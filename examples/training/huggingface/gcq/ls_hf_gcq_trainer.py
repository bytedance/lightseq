import logging
import torch
import torch.distributed as dist
from transformers import Trainer
from packaging import version
from lightseq.training.gcq import (
    GCQState,
    encode_and_decode,
)
from examples.training.huggingface.gcq import GCQArguments

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
        model = super()._wrap_model(model, training, dataloader)
        # Enable GCQ.
        if (
            isinstance(model, torch.nn.parallel.DistributedDataParallel)
            and self.gcq_args.enable_GCQ
        ):
            assert version.parse(torch.__version__) >= version.parse(
                "1.10"
            ), "Training with GCQ requires that the version of torch has to be greater than or equal to 1.10!"
            state = GCQState(
                process_group=dist.group.WORLD if dist.is_initialized() else None,
                hidden_size=self.gcq_args.hidden_size,
                quantile_value=self.gcq_args.GCQ_quantile,
            )
            # Register the communication hook.
            model.register_comm_hook(state=state, hook=encode_and_decode)
            logger.info("############ register communication hook done ###########")

        return model
