import math
import torch
import torch.distributed as dist


class GCQ(object):
    """
    Gradient Communication Quantization (GCQ) in multi-machine distributed training.

    It quantizes gradients to int8 locally before gradients communicate between machines.
    When the gradient communication is done, it dequantizes the int8 gradients to original dtype (default: fp16) locally.

    Note: When training on multiple machines, it can accelerate significantly, as multi-machine communication is the bottleneck.
    However, it is even slower when training on single machine.
    Please choose if GCQ should be on according to your training environment.
    """

    def __init__(
        self,
        world_size,
        hidden_size=1024,
        bucket=None,
        quantize_bits=8,
        quantile_value=0.99,
    ):
        self.world_size = world_size
        self.bucket = bucket
        self.quantize_bits = quantize_bits
        self.device = bucket.buffer().device
        self.hidden_size = hidden_size
        self.quantile_value = quantile_value
        # Actually, we use topk implementation to estimate quantile,
        # as the former is faster than the latter.
        self.topk_value = math.ceil(self.hidden_size * (1.0 - self.quantile_value))

        # We use symmetrical range, i.e., [-127, 127] instead of [-128, 127], when quantize_bits is 8.
        self.max_bound = (
            torch.pow(
                torch.tensor(2.0, dtype=torch.float16, device=self.device),
                self.quantize_bits - 1,
            )
            - 1.0
        )
        self.min_bound = -self.max_bound
        # The new range after dividing the world_size.
        self.new_max_bound = self.max_bound.div(self.world_size, rounding_mode="trunc")
        self.new_min_bound = self.min_bound.div(self.world_size, rounding_mode="trunc")

        self.original_gradient = self.bucket.buffer()
        self.bucket_size = self.original_gradient.numel()
        self.scale = None
        self.encoded_gradient = None

    def get_scale(self):
        """
        Calculate the scale for quantization.
        """
        # Pad original gradient if needed.
        if self.bucket_size % self.hidden_size != 0:
            rest_size = self.hidden_size - (self.bucket_size % self.hidden_size)
            self.original_gradient = torch.nn.functional.pad(
                self.original_gradient, (0, rest_size)
            )
        # Reshape original gradient to [-1, self.hidden_size].
        self.original_gradient = self.original_gradient.view(-1, self.hidden_size)
        # Calculate the max absolute value of every row in original gradient.
        max_abs_bucket = (
            self.original_gradient.abs()
            .topk(self.topk_value, dim=-1, sorted=False)[0]
            .min(dim=-1, keepdim=True)[0]
        )
        self.scale = max_abs_bucket.div(self.max_bound)
        return self.scale

    def encode_bucket(self):
        """
        Quantize gradients to int8 locally before gradients communicate between machines.
        """
        # The original gradient need to divide the world_size to avoid all_reduce overflow.
        compressed_gradient = self.original_gradient.div(self.scale * self.world_size)
        self.encoded_gradient = (
            compressed_gradient.clamp_(min=self.new_min_bound, max=self.new_max_bound)
            .round_()
            .to(torch.int8)
        )
        return self.encoded_gradient

    def decode_bucket(self):
        """
        Dequantize the all_reduced int8 gradients to original dtype (default: fp16) locally.
        """
        decompressed_gradient = self.bucket.buffer()
        # Use original bucket to reduce memory consumption.
        decompressed_gradient.copy_(
            (self.encoded_gradient.to(torch.float16) * self.scale).view(-1)[
                : self.bucket_size
            ]
        )
        return decompressed_gradient


class GCQState(object):
    """
    Prepare the state for Gradient Communication Quantization (GCQ).
    """

    def __init__(self, process_group, hidden_size=1024, quantile_value=0.99):
        self.process_group = process_group
        self.hidden_size = hidden_size
        self.quantile_value = quantile_value


def encode_and_decode(state, bucket) -> torch.futures.Future[torch.Tensor]:
    """
    The Gradient Communication Quantization hook.
    """
    assert (
        state.process_group is not None
    ), "The process_group should be initialized first!"
    process_group = state.process_group
    world_size = dist.get_world_size(process_group)
    hidden_size = state.hidden_size
    quantile_value = state.quantile_value
    quantizer = GCQ(
        world_size=world_size,
        hidden_size=hidden_size,
        bucket=bucket,
        quantile_value=quantile_value,
    )
    scale = quantizer.get_scale()
    # All_reduce scales between ranks and
    # every rank gets the same global scale to encode gradients finally.
    dist.all_reduce(
        scale,
        op=dist.ReduceOp.MAX,
        group=process_group,
        async_op=True,
    ).get_future().wait()
    # Quantize gradients to int8 locally.
    compressed_gradient = quantizer.encode_bucket()
    # All_reduce int8 gradients.
    fut = dist.all_reduce(
        compressed_gradient, group=process_group, async_op=True
    ).get_future()

    def decompress(fut):
        """
        Dequantize the all_reduced int8 gradients to original dtype (default: fp16).
        """
        decompressed_gradient = quantizer.decode_bucket()
        dist.barrier(group=process_group, async_op=True)
        return decompressed_gradient

    return fut.then(decompress)
