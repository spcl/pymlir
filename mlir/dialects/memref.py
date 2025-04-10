""" Implementation of the Memref dialect. """

import inspect
import sys
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

import mlir.astnodes as mast
from mlir.dialect import Dialect, DialectOp, is_op

Literal = Union[mast.StringLiteral, float, int, bool]
SsaUse = Union[mast.SsaId, Literal]

@dataclass
class LoadOperation(DialectOp):
    arg: SsaUse
    index: List[SsaUse]
    type: mast.MemRefType
    _syntax_ = 'memref.load {arg.ssa_use} [ {index.ssa_use_list} ] : {type.memref_type}'

@dataclass
class StoreOperation(DialectOp):
    addr: SsaUse
    ref: SsaUse
    index: List[SsaUse]
    type: mast.MemRefType
    _syntax_ = 'memref.store {addr.ssa_use} , {ref.ssa_use} [ {index.ssa_use_list} ] : {type.memref_type}'

# Inspect current module to get all classes defined above
memref = Dialect('memref', ops=[m[1] for m in inspect.getmembers(
    sys.modules[__name__], lambda obj: is_op(obj, __name__))])
