""" Implementation of the SCF (Structured Control Flow) dialect. """

import inspect
import sys
from mlir.dialect import Dialect, DialectOp, is_op, UnaryOperation
import mlir.astnodes as ast
from dataclasses import dataclass
from typing import Optional


@dataclass
class SCFForOp(DialectOp):
    index: ast.SsaId
    begin: ast.SsaId
    end: ast.SsaId
    body: ast.Region
    step: Optional[ast.SsaId] = None
    _syntax_ = ['scf.for {index.ssa_id} = {begin.ssa_id} to {end.ssa_id} {body.region}',
                'scf.for {index.ssa_id} = {begin.ssa_id} to {end.ssa_id} step {step.ssa_id} {body.region}']


@dataclass
class SCFIfOp(DialectOp):
    cond: ast.SsaId
    body: ast.Region
    elsebody: Optional[ast.Region] = None
    _syntax_ = ['scf.if {cond.ssa_id} {body.region}',
                'scf.if {cond.ssa_id} {body.region} else {elsebody.region}']


class SCFYield(UnaryOperation): _opname_ = 'scf.yield'


# Inspect current module to get all classes defined above
scf = Dialect('scf', ops=[m[1] for m in inspect.getmembers(
               sys.modules[__name__], lambda obj: is_op(obj, __name__))])
