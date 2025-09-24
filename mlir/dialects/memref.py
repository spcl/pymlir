""" Implementation of the Memref dialect. """

import inspect
import sys
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

import mlir.astnodes as mast
from mlir.dialect import Dialect, DialectOp, is_op

Literal = Union[mast.StringLiteral, float, int, bool]
SsaUse = Union[mast.SsaId, Literal]


# class AssumeAlignmentOperation(DialectOp): pass
# class AtomicRMWOperation(DialectOp): pass

@dataclass
class AtomicYieldOperation(DialectOp):
    result: SsaUse
    result_type: mast.Type
    _syntax_ = 'memref.atomic_yield {result.ssa_use} : {result_type.type}'

@dataclass
class CopyOperation(DialectOp):
    source: SsaUse
    target: SsaUse
    source_type: mast.MemRefType
    target_type: mast.MemRefType
    _syntax_ = 'memref.copy {source.ssa_use} , {target.ssa_use} : {source.memref_type} to {target.memref_type}'


# class GenericAtomicRMWOperation(DialectOp): pass

@dataclass
class LoadOperation(DialectOp):
    arg: SsaUse
    index: List[SsaUse]
    type: mast.MemRefType
    _syntax_ = 'memref.load {arg.ssa_use} [ {index.ssa_use_list} ] : {type.memref_type}'


@dataclass
class AllocOperation(DialectOp):
    args: mast.DimAndSymbolList
    type: mast.MemRefType
    _syntax_ = 'memref.alloc {args.dim_and_symbol_use_list} : {type.memref_type}'

@dataclass
class AllocaOperation(DialectOp): 
    args: mast.DimAndSymbolList
    type: mast.MemRefType
    _syntax_ = 'memref.alloca {args.dim_and_symbol_use_list} : {type.memref_type}'

# class AllocaScopeOperation(DialectOp): pass
# class AllocaScopeReturnOperation(DialectOp): pass

@dataclass
class CastOperation(DialectOp):
    arg: SsaUse
    src_type: mast.Type
    dst_type: mast.Type
    _syntax_ = 'memref_cast {arg.ssa_use} : {src_type.type} to {dst_type.type}'

# class CollapseShapeOperation(DialectOp): pass

@dataclass
class DeallocOperation(DialectOp):
    arg: SsaUse
    type: mast.MemRefType
    _syntax_ = 'memref.dealloc {arg.ssa_use} : {type.memref_type}'

@dataclass
class DimOperation(DialectOp):
    operand: mast.SsaId
    index: mast.SsaId
    type: mast.Type
    _syntax_ = 'memref.dim {operand.ssa_id} , {index.ssa_id} : {type.type}'

@dataclass
class DmaStartOperation(DialectOp):
    src: SsaUse
    src_index: List[SsaUse]
    dst: SsaUse
    dst_index: List[SsaUse]
    size: SsaUse
    tag: SsaUse
    tag_index: List[SsaUse]
    src_type: mast.MemRefType
    dst_type: mast.MemRefType
    tag_type: mast.MemRefType
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
    type: mast.MemRefType
    _syntax_ = 'dma_wait {tag.ssa_use} [ {tag_index.ssa_use_list} ] , {size.ssa_use} : {type.memref_type}'

# class ExpandShapeOperation(DialectOp): pass

@dataclass
class ExtractAlignedPointerAsIndexOperation(DialectOp):
    source: SsaUse
    source_type: mast.Type
    dest_type: mast.Type
    _syntax_ = 'memref.extract_aligned_pointer_as_index {source.ssa_use} : {source_type.type} -> {dest_type.type}'

# class ExtractStridedMetadataOperation(DialectOp): pass

@dataclass
class GetGlobalOperation(DialectOp): 
    name: SsaUse
    result_type: mast.MemRefType
    _syntax_ = 'memref.get_global {name.ssa_use} : {result_type.type}'


# class GlobalOperation(DialectOp): pass

@dataclass
class MemorySpaceCastOperation(DialectOp): 
    source: SsaUse
    source_type: mast.MemRefType
    dest_type: mast.MemRefType
    _syntax_ = 'memref.memory_space_cast {source.ssa_use} : {source_type.memref_type} to {dest_type.memref_type}'


# class PrefetchOperation(DialectOp): pass

@dataclass
class RankOperation(DialectOp):
    operand: SsaUse
    op_type: mast.MemRefType
    _syntax_ = 'memref.rank {operand.ssa_use} : {op_type.memref_type}'


# class ReallocOperation(DialectOp): pass
# class ReinterpretCastOperation(DialectOp): pass
# class ReshapeOperation(DialectOp): pass

@dataclass
class StoreOperation(DialectOp):
    addr: SsaUse
    ref: SsaUse
    index: List[SsaUse]
    type: mast.MemRefType
    _syntax_ = 'memref.store {addr.ssa_use} , {ref.ssa_use} [ {index.ssa_use_list} ] : {type.memref_type}'


# class TransposeOperation(DialectOp): pass

@dataclass
class ViewOperation(DialectOp):
    operand: SsaUse
    offset: SsaUse
    src_type: mast.Type
    dst_type: mast.Type
    sizes: Optional[List[SsaUse]] = None
    _syntax_ = ['memref.view {operand.ssa_use} [ {offset.ssa_use} ] [ {sizes.ssa_use_list} ] : {src_type.type} to {dst_type.type}',
                'memref.view {operand.ssa_use} [ {offset.ssa_use} ] [  ] : {src_type.type} to {dst_type.type}']


@dataclass
class SubviewOperation(DialectOp):
    operand: SsaUse
    offsets: List[SsaUse]
    sizes: List[SsaUse]
    strides: List[SsaUse]
    src_type: mast.Type
    dst_type: mast.Type
    _syntax_ = 'memref.subview {operand.ssa_use} [ {offsets.ssa_use_list} ] [ {sizes.ssa_use_list} ] [ {strides.ssa_use_list} ] : {src_type.type} to {dst_type.type}'


# Inspect current module to get all classes defined above
memref = Dialect('memref', ops=[m[1] for m in inspect.getmembers(
    sys.modules[__name__], lambda obj: is_op(obj, __name__))])
