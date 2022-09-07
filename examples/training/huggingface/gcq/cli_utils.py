from dataclasses import dataclass, field


@dataclass
class GCQArguments:
    """
    Arguments Gradient Communication Quantization.
    """

    enable_GCQ: bool = field(default=False, metadata={"help": "Whether to enable GCQ"})
    GCQ_quantile: float = field(
        default=0.99, metadata={"help": "GCQ quantile value, between 0.0-1.0"}
    )
    hidden_size: int = field(
        default=1024, metadata={"help": "The hidden size of model"}
    )
