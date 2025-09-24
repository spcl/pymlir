""" Implementation of the CF (Control Flow) dialect. """

import inspect
import sys
from mlir.dialect import Dialect, DialectOp, is_op, UnaryOperation
import mlir.astnodes as mast
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union

Literal = Union[mast.StringLiteral, float, int, bool]
SsaUse = Union[mast.SsaId, Literal]


@dataclass
class BrOperation(DialectOp):
    block_id: mast.BlockId
    args: Optional[List[Tuple[mast.SsaId, mast.Type]]] = None
    _syntax_ = ['cf.br {block.block_id}',
                'cf.br {block.block_id} {args.block_arg_list}']


@dataclass
class CondBrOperation(DialectOp):
    cond: SsaUse
    block_true: mast.BlockId
    block_false: mast.BlockId
    _syntax_ = ['cf.cond_br {cond.ssa_use} , {block_true.block_id} , {block_false.block_id}']


# Inspect current module to get all classes defined above
cf = Dialect('cf', ops=[m[1] for m in inspect.getmembers(
               sys.modules[__name__], lambda obj: is_op(obj, __name__))])
