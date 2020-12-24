""" Implementation of the Affine dialect. """

import inspect
import sys
import mlir.astnodes as ast
from mlir.dialect import Dialect, DialectOp, is_op
from typing import Union, Optional, List
from dataclasses import dataclass

Literal = Union[ast.StringLiteral, float, int, bool]
SsaUse = Union[ast.SsaId, Literal]


@dataclass
class AffineApplyOp(DialectOp):
    map: ast.AffineMap
    args: ast.DimAndSymbolList
    _syntax_ = 'affine.apply {map.affine_map} {args.dim_and_symbol_use_list}'


@dataclass
class AffineForOp(DialectOp):
    index: ast.SsaId
    begin: Union[ast.SsaId, int]
    end: Union[ast.SsaId, int]
    body: ast.Region
    step: Optional[Union[ast.SsaId, int]] = None
    attributes: Optional[ast.Attribute] = None

    _syntax_ = [
        'affine.for {index.ssa_id} = {begin.symbol_or_const} to {end.symbol_or_const} {body.region}',
        'affine.for {index.ssa_id} = {begin.symbol_or_const} to {end.symbol_or_const} step {step.symbol_or_const} {body.region}',
        'affine.for {index.ssa_id} = {begin.symbol_or_const} to {end.symbol_or_const} {body.region} {attributes.attribute_dict}',
        'affine.for {index.ssa_id} = {begin.symbol_or_const} to {end.symbol_or_const} step {step.symbol_or_const} {body.region} {attributes.attribute_dict}'
    ]


@dataclass
class AffineIfOp(DialectOp):
    cond: ast.MapOrSetId
    operands: List[SsaUse]
    body: ast.Region
    elsebody: Optional[ast.Region] = None

    _syntax_ = ['affine.if {cond.map_or_set_id} ( {operands.ssa_use_list} ) {body.region}',
                'affine.if {cond.map_or_set_id} ( {operands.ssa_use_list} ) {body.region} else {elsebody.region}']


@dataclass
class AffineLoadOp(DialectOp):
    arg: SsaUse
    index: ast.MultiDimAffineExpr
    type: ast.MemRefType
    _syntax_ = 'affine.load {arg.ssa_use} [ {index.multi_dim_affine_expr_no_parens} ] : {type.memref_type}'


@dataclass
class AffineStoreOp(DialectOp):
    addr: SsaUse
    ref: SsaUse
    index: ast.MultiDimAffineExpr
    type: ast.MemRefType
    _syntax_ = 'affine.store {addr.ssa_use} , {ref.ssa_use} [ {index.multi_dim_affine_expr_no_parens} ] : {type.memref_type}'


@dataclass
class AffineMinOp(DialectOp):
    map: ast.AffineMap
    operands: ast.DimAndSymbolList
    _syntax_ = 'affine.min {map.affine_map_inline} {operands.dim_and_symbol_use_list}'


@dataclass
class AffinePrefetchOp(DialectOp):
    arg: SsaUse
    index: ast.MultiDimAffineExpr
    specifier: ast.Identifier
    locality: int
    cachetype: ast.Identifier
    type: ast.Type
    _syntax_ = 'affine.prefetch {arg.ssa_use} [ {index.multi_dim_affine_expr_no_parens} ] , {specifier.bare_id} , locality < {locality.integer_literal} > , {cachetype.bare_id} : {type.type}'


@dataclass
class AffineDmaStartOperation(DialectOp):
    src: SsaUse
    src_index: List[SsaUse]
    dst: SsaUse
    dst_index: List[SsaUse]
    size: SsaUse
    tag: SsaUse
    tag_index: List[SsaUse]
    src_type: ast.MemRefType
    dst_type: ast.MemRefType
    tag_type: ast.MemRefType
    stride: Optional[SsaUse] = None
    transfer_per_stride: Optional[SsaUse] = None

    _syntax_ = [
        'affine.dma_start {src.ssa_use} [ {src_index.multi_dim_affine_expr_no_parens} ] , {dst.ssa_use} [ {dst_index.multi_dim_affine_expr_no_parens} ] , {tag.ssa_use} [ {tag_index.multi_dim_affine_expr_no_parens} ] , {size.ssa_use} : {src_type.memref_type} , {dst_type.memref_type} , {tag_type.memref_type}',
        'affine.dma_start {src.ssa_use} [ {src_index.multi_dim_affine_expr_no_parens} ] , {dst.ssa_use} [ {dst_index.multi_dim_affine_expr_no_parens} ] , {tag.ssa_use} [ {tag_index.multi_dim_affine_expr_no_parens} ] , {size.ssa_use} , {stride.ssa_use} , {transfer_per_stride.ssa_use} : {src_type.memref_type} , {dst_type.memref_type} , {tag_type.memref_type}'
    ]


@dataclass
class AffineDmaWaitOperation(DialectOp):
    tag: SsaUse
    tag_index: ast.MultiDimAffineExpr
    size: SsaUse
    type: ast.MemRefType

    _syntax_ = 'affine.dma_wait {tag.ssa_use} [ {tag_index.multi_dim_affine_expr_no_parens} ] , {size.ssa_use} : {type.memref_type}'


# Inspect current module to get all classes defined above
affine = Dialect('affine', ops=[m[1] for m in inspect.getmembers(
    sys.modules[__name__], lambda obj: is_op(obj, __name__))])
