""" Implementation of the SCF (Structured Control Flow) dialect. """

import inspect
import sys
from mlir.dialect import Dialect, DialectOp, is_op, UnaryOperation
import mlir.astnodes as mast
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class SCFConditionOp(DialectOp):
    condition: mast.SsaId
    args: List[mast.SsaId]
    out_types: List[mast.Type]
    _syntax_ = ['scf.condition ( {condition.ssa_id} ) {args.ssa_id_list} : {out_types.type_list_no_parens}']


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
                'scf.for {index.ssa_id} = {begin.ssa_id} to {end.ssa_id} step {step.ssa_id} iter_args ( {iter_args.argument_assignment_list_no_parens} ) -> {iter_args_types.type_list_no_parens} {body.region}',
                'scf.for {index.ssa_id} = {begin.ssa_id} to {end.ssa_id} step {step.ssa_id} iter_args ( {iter_args.argument_assignment_list_no_parens} ) -> {iter_args_types.type_list_no_parens} : {out_type.type} {body.region}',
                'scf.for {index.ssa_id} = {begin.ssa_id} to {end.ssa_id} step {step.ssa_id} iter_args ( {iter_args.argument_assignment_list_no_parens} ) -> ( {iter_args_types.type_list_no_parens} ) {body.region}',
                'scf.for {index.ssa_id} = {begin.ssa_id} to {end.ssa_id} step {step.ssa_id} iter_args ( {iter_args.argument_assignment_list_no_parens} ) -> ( {iter_args_types.type_list_no_parens} ) : {out_type.type} {body.region}']


@dataclass
class SCFIfOp(DialectOp):
    cond: mast.SsaId
    body: mast.Region
    elsebody: Optional[mast.Region] = None
    out_types: Optional[List[mast.Type]] = None
    _syntax_ = ['scf.if {cond.ssa_id} {body.region}',
                'scf.if {cond.ssa_id} {body.region} else {elsebody.region}',
                'scf.if {cond.ssa_id} -> ( {out_types.type_list_no_parens} ) {body.region}',
                'scf.if {cond.ssa_id} -> ( {out_types.type_list_no_parens} ) {body.region} else {elsebody.region}']


@dataclass
class SCFWhileOp(DialectOp):
    assignments: List[Tuple[mast.SsaId, mast.Type]]
    out_type: mast.FunctionType
    while_body: mast.Region
    do_body: mast.Region
    _syntax_ = ['scf.while ( {assignments.argument_assignment_list_no_parens} ) : {out_type.function_type} {while_body.region} do {do_body.region}']


@dataclass
class SCFYield(DialectOp):
    results: Optional[List[mast.SsaId]] = None
    result_types: Optional[List[mast.Type]] = None
    _syntax_ = ['scf.yield',
                'scf.yield {results.ssa_id_list} : {result_types.type_list_no_parens}']


# Inspect current module to get all classes defined above
scf = Dialect('scf', ops=[m[1] for m in inspect.getmembers(
               sys.modules[__name__], lambda obj: is_op(obj, __name__))])
