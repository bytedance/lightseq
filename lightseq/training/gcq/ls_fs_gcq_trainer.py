import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from fairseq.trainer import Trainer
from .gcq import GCQState, encode_and_decode

logger = logging.getLogger(__name__)


class LSTrainer(Trainer):
    """
    Main class for data parallel.

    This class supports GCQ (Gradient Communication Quantization) for 
    distributed multi-machine training based on fairseq.trainer.Trainer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def model(self):
        if self._wrapped_model is None:
            super().model
            if (
                isinstance(self._wrapped_model, DistributedDataParallel)
                and self.args.enable_GCQ
            ):
                state = GCQState(
                    process_group=dist.group.WORLD if dist.is_initialized() else None,
                    hidden_size=self.args.encoder_embed_dim,
                    quantile_value=self.args.GCQ_quantile,
                )
                # Register the communication hook.
                self._wrapped_model.register_comm_hook(
                    state=state, hook=encode_and_decode
                )
                logger.info("############ register communication hook done ###########")
        return self._wrapped_model