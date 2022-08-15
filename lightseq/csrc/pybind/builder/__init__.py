from .builder import CUDAOpBuilder
from .operator_builder import OperatorBuilder
from .layer_builder import TransformerBuilder

# TODO: infer this list instead of hard coded
# List of all available ops
__op_builders__ = [OperatorBuilder(), TransformerBuilder()]
ALL_OPS = {op.name: op for op in __op_builders__}
