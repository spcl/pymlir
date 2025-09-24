""" Implementation of the arith (Arithmetic) dialect. """

import inspect
import sys
from mlir.dialect import Dialect, DialectOp, is_op, UnaryOperation, BinaryOperation
import mlir.astnodes as mast
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union

Literal = Union[mast.StringLiteral, float, int, bool]
SsaUse = Union[mast.SsaId, Literal]


# Unary Operations

class BitcastOperation(UnaryOperation): _opname_ = 'arith.bitcast'
class ExtFOperation(UnaryOperation): _opname_ = 'arith.extf'
class ExtSIOperation(UnaryOperation): _opname_ = 'arith.extsi'
class ExtUIOperation(UnaryOperation): _opname_ = 'arith.extui'
class FPToSIOperation(UnaryOperation): _opname_ = 'arith.fptosi'
class FPToUIOperation(UnaryOperation): _opname_ = 'arith.fptoui'
class NegFOperation(UnaryOperation): _opname_ = 'arith.negf'
class SIToFPOperation(UnaryOperation): _opname_ = 'arith.sitofp'
class UIToFPOperation(UnaryOperation): _opname_ = 'arith.uitofp'

# Arithmetic Operations
class AddFOperation(BinaryOperation): _opname_ = 'arith.addf'
class AddIOperation(BinaryOperation): _opname_ = 'arith.addi'
class AndIOperation(BinaryOperation): _opname_ = 'arith.andi'
class CeilDivSIOperation(BinaryOperation): _opname_ = 'arith.ceildivsi'
class CeilDivUIOperation(BinaryOperation): _opname_ = 'arith.ceildivui'
class DivFOperation(BinaryOperation): _opname_ = 'arith.divf'
class DivSIOperation(BinaryOperation): _opname_ = 'arith.divsi'
class DivUIOperation(BinaryOperation): _opname_ = 'arith.divui'
class FloorDivSIOperation(BinaryOperation): _opname_ = 'arith.floordivsi'
class MaximumFOperation(BinaryOperation): _opname_ = 'arith.maximumf'
class MaxNumFOperation(BinaryOperation): _opname_ = 'arith.maxnumf'
class MaxSIOperation(BinaryOperation): _opname_ = 'arith.maxsi'
class MaxUIOperation(BinaryOperation): _opname_ = 'arith.maxui'
class MinimumFOperation(BinaryOperation): _opname_ = 'arith.minimumf'
class MinNumFOperation(BinaryOperation): _opname_ = 'arith.minnumf'
class MinSIOperation(BinaryOperation): _opname_ = 'arith.minsi'
class MinUIOperation(BinaryOperation): _opname_ = 'arith.minui'
class MulFOperation(BinaryOperation): _opname_ = 'arith.mulf'
class MulIOperation(BinaryOperation): _opname_ = 'arith.muli'
class MulSIExtendedOp(BinaryOperation): _opname_ = 'arith.mulsi_extended'
class MulUIExtendedOp(BinaryOperation): _opname_ = 'arith.mului_extended'
class OrIOperation(BinaryOperation): _opname_ = 'arith.ori'
class RemFOperation(BinaryOperation): _opname_ = 'arith.remf'
class RemSIOperation(BinaryOperation): _opname_ = 'arith.remsi'
class RemUIOperation(BinaryOperation): _opname_ = 'arith.remui'
class ShLIOperation(BinaryOperation): _opname_ = 'arith.shli'
class ShRSIOperation(BinaryOperation): _opname_ = 'arith.shrsi'
class ShRUIOperation(BinaryOperation): _opname_ = 'arith.shrui'
class SubIOperation(BinaryOperation): _opname_ = 'arith.subi'
class SubFOperation(BinaryOperation): _opname_ = 'arith.subf'
class TruncFOperation(BinaryOperation): _opname_ = 'arith.truncf'
class TruncIOperation(BinaryOperation): _opname_ = 'arith.trunci'
class XorIOperation(BinaryOperation): _opname_ = 'arith.xori'


@dataclass
class AddUIExtendedOperation(DialectOp):
    lhs_operand: mast.SsaId
    rhs_operand: mast.SsaId
    sum_type: mast.Type
    ovf_type = mast.Type
    _syntax_ = 'arith.addui_extended {lhs_operand.ssa_id} , {rhs_operand.ssa_id} : {sum_type.type} , {ovf_type.type}'


@dataclass
class CmpiOperation(DialectOp):
    comptype: str
    operand_a: mast.SsaId
    operand_b: mast.SsaId
    type: mast.Type
    _syntax_ = 'arith.cmpi {comptype.string_literal} , {operand_a.ssa_id} , {operand_b.ssa_id} : {type.type}'


@dataclass
class CmpfOperation(DialectOp):
    comptype: str
    operand_a: mast.SsaId
    operand_b: mast.SsaId
    type: mast.Type
    _syntax_ = 'arith.cmpf {comptype.string_literal} , {operand_a.ssa_id} , {operand_b.ssa_id} : {type.type}'


@dataclass
class ConstantOperation(DialectOp):
    value: Literal
    type: mast.Type
    _syntax_ = ['arith.constant {value.constant_literal} : {type.type}', 'arith.constant {value.constant_literal}']



@dataclass
class IndexCastOperation(DialectOp):
    arg: SsaUse
    src_type: mast.Type
    dst_type: mast.Type
    _syntax_ = 'arith.index_cast {arg.ssa_use} : {src_type.type} to {dst_type.type}'

@dataclass
class IndexCastUIOperation(DialectOp):
    arg: SsaUse
    src_type: mast.Type
    dst_type: mast.Type
    _syntax_ = 'arith.index_castui {arg.ssa_use} : {src_type.type} to {dst_type.type}'

@dataclass
class SelectOperation(DialectOp):
    cond: SsaUse
    arg_true: SsaUse
    arg_false: SsaUse
    _syntax_ = 'arith.select {cond.ssa_use} , {arg_true.ssa_use} , {arg_false.ssa_use} : {type.type}'


# Inspect current module to get all classes defined above
arith = Dialect('arith', ops=[m[1] for m in inspect.getmembers(
               sys.modules[__name__], lambda obj: is_op(obj, __name__))])
