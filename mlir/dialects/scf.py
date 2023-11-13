""" Implementation of the SCF (Structured Control Flow) dialect. """

import inspect
import sys
from mlir.dialect import Dialect, DialectOp, is_op, UnaryOperation
import mlir.astnodes as mast
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class SCFForOp(DialectOp):
    index: mast.SsaId
    begin: mast.SsaId
    end: mast.SsaId
    step: mast.SsaId
    body: mast.Region
    iter_args: Optional[List[Tuple[mast.SsaId, mast.SsaId]]] = None
    iter_args_types: Optional[List[mast.Type]] = None
    out_type: Optional[mast.Type] = None
    _syntax_ = ['scf.for {index.ssa_id} = {begin.ssa_id} to {end.ssa_id} step {step.ssa_id} {body.region}',
                'scf.for {index.ssa_id} = {begin.ssa_id} to {end.ssa_id} step {step.ssa_id} : {out_type.type} {body.region}',
                'scf.for {index.ssa_id} = {begin.ssa_id} to {end.ssa_id} step {step.ssa_id} iter_args {iter_args.argument_assignment_list_parens} -> {iter_args_types.type_list_parens} {body.region}',
                'scf.for {index.ssa_id} = {begin.ssa_id} to {end.ssa_id} step {step.ssa_id} iter_args {iter_args.argument_assignment_list_parens} -> {iter_args_types.type_list_parens} : {out_type.type} {body.region}']


@dataclass
class SCFIfOp(DialectOp):
    cond: mast.SsaId
    body: mast.Region
    elsebody: Optional[mast.Region] = None
    _syntax_ = ['scf.if {cond.ssa_id} {body.region}',
                'scf.if {cond.ssa_id} {body.region} else {elsebody.region}']


class SCFYield(UnaryOperation): _opname_ = 'scf.yield'


# Inspect current module to get all classes defined above
scf = Dialect('scf', ops=[m[1] for m in inspect.getmembers(
               sys.modules[__name__], lambda obj: is_op(obj, __name__))])
