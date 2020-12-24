""" Implementation of the Standard dialect. """

import inspect
import sys
from mlir.dialect import (Dialect, DialectOp, UnaryOperation, BinaryOperation,
                          is_op)
import mlir.astnodes as ast
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass


Literal = Union[ast.StringLiteral, float, int, bool]
SsaUse = Union[ast.SsaId, Literal]


# Terminator Operations
@dataclass
class BrOperation(DialectOp):
    block_id: ast.BlockId
    args: Optional[List[Tuple[ast.SsaId, ast.Type]]] = None
    _syntax_ = ['br {block.block_id}',
                'br {block.block_id} {args.block_arg_list}']


@dataclass
class CondBrOperation(DialectOp):
    cond: SsaUse
    block_true: ast.BlockId
    block_false: ast.BlockId
    _syntax_ = ['cond_br {cond.ssa_use} , {block_true.block_id} , {block_false.block_id}']


@dataclass
class ReturnOperation(DialectOp):
    values: Optional[List[SsaUse]] = None
    types: Optional[List[ast.Type]] = None
    _syntax_ = ['return',
                'return {values.ssa_use_list} : {types.type_list_no_parens}']


# Core Operations
@dataclass
class CallOperation(DialectOp):
    func: ast.SymbolRefId
    type: ast.FunctionType
    args: Optional[List[SsaUse]] = None
    argtypes: Optional[List[ast.Type]] = None
    _syntax_ = ['call {func.symbol_ref_id} () : {type.function_type}',
                'call {func.symbol_ref_id} ( {args.ssa_use_list} ) : {argtypes.function_type}']


@dataclass
class CallIndirectOperation(DialectOp):
    func: ast.SymbolRefId
    type: ast.FunctionType
    args: Optional[List[SsaUse]] = None
    argtypes: Optional[List[ast.Type]] = None
    _syntax_ = ['call_indirect {func.symbol_ref_id} () : {type.function_type}',
                'call_indirect {func.symbol_ref_id} ( {args.ssa_use_list} ) : {type.function_type}']


@dataclass
class DimOperation(DialectOp):
    operand: ast.SsaId
    index: ast.SsaId
    type: ast.Type
    _syntax_ = 'dim {operand.ssa_id} , {index.ssa_id} : {type.type}'


# Memory Operations
@dataclass
class AllocOperation(DialectOp):
    args: ast.DimAndSymbolList
    type: ast.MemRefType
    _syntax_ = 'alloc {args.dim_and_symbol_use_list} : {type.memref_type}'


@dataclass
class AllocStaticOperation(DialectOp):
    base: int
    type: ast.MemRefType
    _syntax_ = 'alloc_static ( {base.integer_literal} ) : {type.memref_type}'


@dataclass
class DeallocOperation(DialectOp):
    arg: SsaUse
    type: ast.MemRefType
    _syntax_ = 'dealloc {arg.ssa_use} : {type.memref_type}'


@dataclass
class DmaStartOperation(DialectOp):
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
        'dma_start {src.ssa_use} [ {src_index.ssa_use_list} ] , {dst.ssa_use} [ {dst_index.ssa_use_list} ] , {size.ssa_use} , {tag.ssa_use} [ {tag_index.ssa_use_list} ] : {src_type.memref_type} , {dst_type.memref_type} , {tag_type.memref_type}',
        'dma_start {src.ssa_use} [ {src_index.ssa_use_list} ] , {dst.ssa_use} [ {dst_index.ssa_use_list} ] , {size.ssa_use} , {tag.ssa_use} [ {tag_index.ssa_use_list} ] , {stride.ssa_use} , {transfer_per_stride.ssa_use} : {src_type.memref_type} , {dst_type.memref_type} , {tag_type.memref_type}'
    ]
@dataclass
class DmaWaitOperation(DialectOp):
    tag: SsaUse
    tag_index: List[SsaUse]
    size: SsaUse
    type: ast.MemRefType
    _syntax_ = 'dma_wait {tag.ssa_use} [ {tag_index.ssa_use_list} ] , {size.ssa_use} : {type.memref_type}'


@dataclass
class ExtractElementOperation(DialectOp):
    arg: SsaUse
    index: List[SsaUse]
    type: ast.Type
    _syntax_ = 'extract_element {arg.ssa_use} [ {index.ssa_use_list} ] : {type.type}'


@dataclass
class LoadOperation(DialectOp):
    arg: SsaUse
    index: List[SsaUse]
    type: ast.MemRefType
    _syntax_ = 'load {arg.ssa_use} [ {index.ssa_use_list} ] : {type.memref_type}'


@dataclass
class SplatOperation(DialectOp):
    arg: SsaUse
    type: Union[ast.VectorType, ast.TensorType]
    _syntax_ = 'splat {arg.ssa_use} : {type.type}'  # (vector_type | tensor_type)


@dataclass
class StoreOperation(DialectOp):
    addr: SsaUse
    ref: SsaUse
    index: List[SsaUse]
    type: ast.MemRefType
    _syntax_ = 'store {addr.ssa_use} , {ref.ssa_use} [  {index.ssa_use_list} ] : {type.memref_type}'


@dataclass
class TensorLoadOperation(DialectOp):
    arg: SsaUse
    type: ast.Type
    _syntax_ = 'tensor_load {arg.ssa_use} : {type.type}'


@dataclass
class TensorStoreOperation(DialectOp):
    arg: SsaUse
    type: ast.Type
    _syntax_ = 'tensor_store {src.ssa_use} , {dst.ssa_use} : {type.memref_type}'

# Unary Operations
class AbsfOperation(UnaryOperation): _opname_ = 'absf'
class CeilfOperation(UnaryOperation): _opname_ = 'ceilf'
class CosOperation(UnaryOperation): _opname_ = 'cos'
class ExpOperation(UnaryOperation): _opname_ = 'exp'
class NegfOperation(UnaryOperation): _opname_ = 'negf'
class TanhOperation(UnaryOperation): _opname_ = 'tanh'
class CopysignOperation(UnaryOperation): _opname_ = 'copysign'
class SIToFPOperation(UnaryOperation): _opname_ = 'sitofp'

# Arithmetic Operations
class AddiOperation(BinaryOperation): _opname_ = 'addi'
class AddfOperation(BinaryOperation): _opname_ = 'addf'
class AndOperation(BinaryOperation): _opname_ = 'and'
class DivisOperation(BinaryOperation): _opname_ = 'divis'
class DiviuOperation(BinaryOperation): _opname_ = 'diviu'
class RemisOperation(BinaryOperation): _opname_ = 'remis'
class RemiuOperation(BinaryOperation): _opname_ = 'remiu'
class DivfOperation(BinaryOperation): _opname_ = 'divf'
class MulfOperation(BinaryOperation): _opname_ = 'mulf'
class MulIOperation(BinaryOperation): _opname_ = 'muli'
class SubiOperation(BinaryOperation): _opname_ = 'subi'
class SubfOperation(BinaryOperation): _opname_ = 'subf'
class OrOperation(BinaryOperation): _opname_ = 'or'
class XorOperation(BinaryOperation): _opname_ = 'xor'


@dataclass
class CmpiOperation(DialectOp):
    comptype: str
    operand_a: ast.SsaId
    operand_b: ast.SsaId
    type: ast.Type
    _syntax_ = 'cmpi {comptype.string_literal} , {operand_a.ssa_id} , {operand_b.ssa_id} : {type.type}'


@dataclass
class CmpfOperation(DialectOp):
    comptype: str
    operand_a: ast.SsaId
    operand_b: ast.SsaId
    type: ast.Type
    _syntax_ = 'cmpf {comptype.string_literal} , {operand_a.ssa_id} , {operand_b.ssa_id} : {type.type}'


@dataclass
class ConstantOperation(DialectOp):
    value: Literal
    type: ast.Type
    _syntax_ = 'constant {value.constant_literal} : {type.type}'


@dataclass
class IndexCastOperation(DialectOp):
    arg: SsaUse
    src_type: ast.Type
    dst_type: ast.Type
    _syntax_ = 'index_cast {arg.ssa_use} : {src_type.type} to {dst_type.type}'


@dataclass
class MemrefCastOperation(DialectOp):
    arg: SsaUse
    src_type: ast.Type
    dst_type: ast.Type
    _syntax_ = 'memref_cast {arg.ssa_use} : {src_type.type} to {dst_type.type}'


@dataclass
class TensorCastOperation(DialectOp):
    arg: SsaUse
    src_type: ast.Type
    dst_type: ast.Type
    _syntax_ = 'tensor_cast {arg.ssa_use} : {src_type.type} to {dst_type.type}'


@dataclass
class SelectOperation(DialectOp):
    cond: SsaUse
    arg_true: SsaUse
    arg_false: SsaUse
    _syntax_ = 'select {cond.ssa_use} , {arg_true.ssa_use} , {arg_false.ssa_use} : {type.type}'


@dataclass
class SubviewOperation(DialectOp):
    operand: SsaUse
    offsets: List[SsaUse]
    sizes: List[SsaUse]
    strides: List[SsaUse]
    src_type: ast.Type
    dst_type: ast.Type
    _syntax_ = 'subview {operand.ssa_use} [ {offsets.ssa_use_list} ] [ {sizes.ssa_use_list} ] [ {strides.ssa_use_list} ] : {src_type.type} to {dst_type.type}'


@dataclass
class ViewOperation(DialectOp):
    operand: SsaUse
    offset: SsaUse
    src_type: ast.Type
    dst_type: ast.Type
    sizes: Optional[List[SsaUse]] = None
    _syntax_ = ['view {operand.ssa_use} [ {offset.ssa_use} ] [ {sizes.ssa_use_list} ] : {src_type.type} to {dst_type.type}',
                'view {operand.ssa_use} [ {offset.ssa_use} ] [  ] : {src_type.type} to {dst_type.type}']


# Inspect current module to get all classes defined above
standard = Dialect('standard', ops=[m[1] for m in inspect.getmembers(
    sys.modules[__name__], lambda obj: is_op(obj, __name__))])
