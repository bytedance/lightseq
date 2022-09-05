import functools
import os
import logging

import torch
import torch.distributed as dist
from torch import nn
from packaging import version

from transformers.integrations import is_fairscale_available
from transformers import Trainer
from transformers import __version__
from transformers.dependency_versions_check import dep_version_check
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.trainer_pt_utils import get_module_class_from_name
from transformers.trainer_utils import (
    FSDPOption,
    ShardedDDPOption,
)
from transformers.utils import (
    is_apex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
)

from lightseq.training.gcq import (
    GCQState,
    encode_and_decode,
)
from examples.training.huggingface.gcq import GCQArguments

if is_apex_available():
    from apex import amp

if is_fairscale_available():
    dep_version_check("fairscale")
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION
    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


logger = logging.getLogger("lightseq_hf_trainer")

class LSTrainer(Trainer):
    """
    LSTrainer supports GCQ (Gradient Communication Quantization) for distributed multi-machine training 
    based on transformers.Trainer.
    """
    def __init__(self, gcq_args: GCQArguments = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.setLevel(logging.INFO if self.args.should_log else logging.WARN)
        self.gcq_args = gcq_args
        
    def _wrap_model(self, model, training=True, dataloader=None):
        if self.args.use_ipex:
            dtype = torch.bfloat16 if self.use_cpu_amp else torch.float32
            model = self.ipex_optimize_model(model, training, dtype=dtype)

        if self.args.jit_mode_eval:
            model = self.torch_jit_model_eval(model, dataloader, training)

        if is_sagemaker_mp_enabled():
            # Wrapping the base model twice in a DistributedModel will raise an error.
            if isinstance(self.model_wrapped, smp.model.DistributedModel):
                return self.model_wrapped
            return smp.DistributedModel(
                model, backward_passes_per_step=self.args.gradient_accumulation_steps
            )

        # already initialized its own DDP and AMP
        if self.deepspeed:
            return self.deepspeed

        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if unwrap_model(model) is not model:
            return model

        # Mixed precision training with apex (torch < 1.6)
        if self.use_apex and training:
            model, self.optimizer = amp.initialize(
                model, self.optimizer, opt_level=self.args.fp16_opt_level
            )

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = nn.DataParallel(model)

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return model

        # Distributed training (should be after apex fp16 initialization)
        if self.sharded_ddp is not None:
            # Sharded DDP!
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                model = ShardedDDP(model, self.optimizer)
            else:
                mixed_precision = self.args.fp16 or self.args.bf16
                cpu_offload = ShardedDDPOption.OFFLOAD in self.args.sharded_ddp
                zero_3 = self.sharded_ddp == ShardedDDPOption.ZERO_DP_3
                # XXX: Breaking the self.model convention but I see no way around it for now.
                if ShardedDDPOption.AUTO_WRAP in self.args.sharded_ddp:
                    model = auto_wrap(model)
                self.model = model = FullyShardedDDP(
                    model,
                    mixed_precision=mixed_precision,
                    reshard_after_forward=zero_3,
                    cpu_offload=cpu_offload,
                ).to(self.args.device)

        # Distributed training using PyTorch FSDP
        if self.fsdp is not None:
            # PyTorch FSDP!
            from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
            from torch.distributed.fsdp.fully_sharded_data_parallel import (
                FullyShardedDataParallel as FSDP,
            )
            from torch.distributed.fsdp.fully_sharded_data_parallel import (
                MixedPrecision,
            )
            from torch.distributed.fsdp.wrap import (
                size_based_auto_wrap_policy,
                transformer_auto_wrap_policy,
            )

            if FSDPOption.OFFLOAD in self.args.fsdp:
                cpu_offload = CPUOffload(offload_params=True)
            else:
                cpu_offload = CPUOffload(offload_params=False)

            auto_wrap_policy = None
            if FSDPOption.AUTO_WRAP in self.args.fsdp:
                if self.args.fsdp_min_num_params > 0:
                    auto_wrap_policy = functools.partial(
                        size_based_auto_wrap_policy,
                        min_num_params=self.args.fsdp_min_num_params,
                    )
                elif self.args.fsdp_transformer_layer_cls_to_wrap is not None:
                    transformer_cls_to_wrap = get_module_class_from_name(
                        model, self.args.fsdp_transformer_layer_cls_to_wrap
                    )
                    auto_wrap_policy = functools.partial(
                        transformer_auto_wrap_policy,
                        # Transformer layer class to wrap
                        transformer_layer_cls={transformer_cls_to_wrap},
                    )
            mixed_precision_policy = None
            dtype = None
            if self.args.fp16:
                dtype = torch.float16
            elif self.args.bf16:
                dtype = torch.bfloat16
            if dtype is not None:
                mixed_precision_policy = MixedPrecision(
                    param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype
                )
            if type(model) != FSDP:
                # XXX: Breaking the self.model convention but I see no way around it for now.
                self.model = model = FSDP(
                    model,
                    sharding_strategy=self.fsdp,
                    cpu_offload=cpu_offload,
                    auto_wrap_policy=auto_wrap_policy,
                    mixed_precision=mixed_precision_policy,
                )
                if FSDPOption.OFFLOAD not in self.args.fsdp:
                    model.to(self.args.device)

        elif is_sagemaker_dp_enabled():
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[int(os.getenv("SMDATAPARALLEL_LOCAL_RANK"))]
            )
        elif self.args.local_rank != -1:
            kwargs = {}
            if self.args.ddp_find_unused_parameters is not None:
                kwargs["find_unused_parameters"] = self.args.ddp_find_unused_parameters
            elif isinstance(model, PreTrainedModel):
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                kwargs["find_unused_parameters"] = not model.is_gradient_checkpointing
            else:
                kwargs["find_unused_parameters"] = True

            if self.args.ddp_bucket_cap_mb is not None:
                kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank] if self.args._n_gpu != 0 else None,
                output_device=self.args.local_rank if self.args._n_gpu != 0 else None,
                **kwargs,
            )

        # Enable GCQ.
        if (
            isinstance(model, torch.nn.parallel.DistributedDataParallel)
            and self.gcq_args.enable_GCQ
        ):
            state = GCQState(
                process_group=dist.group.WORLD if dist.is_initialized() else None,
                hidden_size=self.gcq_args.hidden_size,
                quantile_value=self.gcq_args.GCQ_quantile,
            )
            # Register the communication hook.
            model.register_comm_hook(state=state, hook=encode_and_decode)
            logger.info("############ register communication hook done ###########")

        return model

    