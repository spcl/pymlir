""" Implementation of the Tensor dialect. """

import inspect
import sys
from mlir.dialect import Dialect, DialectOp, is_op, UnaryOperation
import mlir.astnodes as mast
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union

Literal = Union[mast.StringLiteral, float, int, bool]
SsaUse = Union[mast.SsaId, Literal]

@dataclass
class ExtractElementOperation(DialectOp):
    arg: SsaUse
    index: List[SsaUse]
    type: mast.Type
    _syntax_ = 'tensor.extract {arg.ssa_use} [ {index.ssa_use_list} ] : {type.type}'


@dataclass
class SplatOperation(DialectOp):
    arg: SsaUse
    type: Union[mast.VectorType, mast.TensorType]
    _syntax_ = 'tensor.splat {arg.ssa_use} : {type.type}'  # (vector_type | tensor_type)

@dataclass
class TensorCastOperation(DialectOp):
    arg: SsaUse
    src_type: mast.Type
    dst_type: mast.Type
    _syntax_ = 'tensor.cast {arg.ssa_use} : {src_type.type} to {dst_type.type}'

# Inspect current module to get all classes defined above
tensor = Dialect('tensor', ops=[m[1] for m in inspect.getmembers(
               sys.modules[__name__], lambda obj: is_op(obj, __name__))])
