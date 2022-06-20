""" Classes containing MLIR AST node types, fields, and conversion back to
    MLIR. """

from enum import Enum, auto
from typing import Any, List, Union, Optional
from lark import Token
from lark.tree import Tree
from dataclasses import dataclass, field, is_dataclass


class Node(object):
    """ Base MLIR AST node object. """
    @classmethod
    def from_lark(cls, args: List[Any]):
        assert isinstance(args, list)
        # ensure that no lark objects creep into our AST
        assert not any(isinstance(el, (Token, Tree)) for el in args)
        return cls(*args)

    def dump_ast(self) -> str:
        from warnings import warn
        warn("dump_ast is deprecated, simply call 'repr' on the object",
             DeprecationWarning, stacklevel=2)
        return repr(self)

    @property
    def _fields_(self) -> List[str]:
        if is_dataclass(self):
            return self.__dataclass_fields__.keys()
        else:
            raise AttributeError(f"'{self.__class__}' object has not attribute '_fields_'")

    def dump(self, indent: int = 0) -> str:
        """ Dumps the AST node and its children in MLIR format.
            :return: String representing the AST in MLIR.
        """
        raise NotImplementedError

    def pretty(self):
        return self.dump()


@dataclass
class StringLiteral(Node):
    value: str

    def dump(self, indent: int = 0):
        return '"%s"' % self.value

    def __str__(self):
        return self.dump()


##############################################################################
# Identifiers

@dataclass
class Identifier(Node):
    value: str
    _prefix_: str = field(init=False, default='', repr=False)

    def dump(self, indent: int = 0) -> str:
        return self._prefix_ + self.value


@dataclass
class SsaId(Identifier):
    value: str
    op_no: Optional[int] = None
    _prefix_: str = field(init=False, default='%', repr=False)

    def dump(self, indent: int = 0) -> str:
        if self.op_no:
            return self._prefix_ + self.value + ('#%s' % self.op_no)
        return self._prefix_ + self.value


class SymbolRefId(Identifier):
    _prefix_ = '@'


class BlockId(Identifier):
    _prefix_ = '^'


class TypeAlias(Identifier):
    _prefix_ = '!'


class AttrAlias(Identifier):
    _prefix_ = '#'


class MapOrSetId(Identifier):
    _prefix_ = '#'


##############################################################################
# Types


class Type(Node):
    pass


@dataclass
class Dimension(Type):
    value: Optional[int] = None

    def dump(self, indent: int = 0) -> str:
        return "?" if self.value is None else str(self.value)


@dataclass
class NoneType(Type):
    def dump(self, indent: int = 0) -> str:
        return 'none'


class FloatTypeEnum(Enum):
    f16 = "f16"
    bf16 = "bf16"
    f32 = "f32"
    f64 = "f64"


@dataclass
class FloatType(Type):
    type: FloatTypeEnum

    def dump(self, indent: int = 0) -> str:
        return self.type.name


@dataclass
class IndexType(Type):
    def dump(self, indent: int = 0) -> str:
        return 'index'


@dataclass
class IntegerType(Type):
    width: int


@dataclass
class SignedIntegerType(IntegerType):
    def dump(self, indent: int = 0) -> str:
        return 'si' + str(self.width)


@dataclass
class UnsignedIntegerType(IntegerType):
    def dump(self, indent: int = 0) -> str:
        return 'ui' + str(self.width)


@dataclass
class SignlessIntegerType(IntegerType):
    def dump(self, indent: int = 0) -> str:
        return 'i' + str(self.width)


@dataclass
class ComplexType(Type):
    type: Type

    def dump(self, indent: int = 0) -> str:
        return 'complex<%s>' % self.type.dump(indent)


@dataclass
class TupleType(Type):
    types: List[Type]

    def dump(self, indent: int = 0) -> str:
        return 'tuple<%s>' % dump_or_value(self.types, indent)


@dataclass
class VectorType(Type):
    dimensions: int
    element_type: Union[IntegerType, FloatType]

    def dump(self, indent: int = 0) -> str:
        return 'vector<%s>' % ('x'.join(
            dump_or_value(t, indent)
            for t in self.dimensions) + 'x' + self.element_type.dump(indent))


class TensorType(Type):
    pass


@dataclass
class RankedTensorType(TensorType):
    dimensions: List[Dimension]
    element_type: Union[IntegerType, FloatType, ComplexType, VectorType]

    def dump(self, indent: int = 0) -> str:
        return 'tensor<%s>' % ('x'.join(
            t.dump(indent)
            for t in self.dimensions) + 'x' + self.element_type.dump(indent))


@dataclass
class UnrankedTensorType(TensorType):
    element_type: Union[IntegerType, FloatType, ComplexType, VectorType]

    def dump(self, indent: int = 0) -> str:
        return 'tensor<*x%s>' % self.element_type.dump(indent)


class MemRefType(Type):
    pass


@dataclass
class StridedLayout(Node):
    offset: int = 0
    strides: Optional[List[int]] = None

    def dump(self, indent: int = 0) -> str:
        return 'offset: %s, strides: [%s]' % (dump_or_value(
            self.offset, indent), dump_or_value(self.strides, indent))


@dataclass
class RankedMemRefType(MemRefType):
    dimensions: List[Dimension]
    element_type: Union[IntegerType, FloatType, ComplexType, VectorType]
    layout: Optional[StridedLayout] = None
    space: Optional[int] = None

    def dump(self, indent: int = 0) -> str:
        result = 'memref<%s' % ('x'.join(t.dump(indent)
                                         for t in self.dimensions)
                                + ('x' if self.dimensions else '')
                                + self.element_type.dump(indent))
        if self.layout is not None:
            result += ', ' + self.layout.dump(indent)
        if self.space is not None:
            result += ', ' + dump_or_value(self.space, indent)

        return result + '>'


@dataclass
class UnrankedMemRefType(MemRefType):
    element_type: Union[IntegerType, FloatType, ComplexType, VectorType]
    space: Optional[int] = None

    def dump(self, indent: int = 0) -> str:
        result = 'memref<%s' % ('*x' + self.element_type.dump(indent))
        if self.space is not None:
            result += ', ' + dump_or_value(self.space, indent)

        return result + '>'


@dataclass
class OpaqueDialectType(Type):
    dialect: str
    contents: str

    def dump(self, indent: int = 0) -> str:
        return '!%s<"%s">' % (self.dialect, self.contents)

@dataclass
class PrettyDialectType(Type):
    dialect: str
    type: str
    body: List[str]

    def dump(self, indent: int = 0) -> str:
        return '!%s.%s<%s>' % (self.dialect, self.type, ', '.join(
            dump_or_value(item, indent) for item in self.body))


@dataclass
class FunctionType(Type):
    argument_types: List[Type]
    result_types: List[Type]

    def dump(self, indent: int = 0) -> str:
        result = '(%s)' % ', '.join(
            dump_or_value(arg, indent) for arg in self.argument_types)
        result += ' -> '
        if not self.result_types:
            result += '()'
        elif len(self.result_types) == 1:
            result += dump_or_value(self.result_types[0], indent)
        else:
            result += '(%s)' % ', '.join(
                dump_or_value(res, indent) for res in self.result_types)
        return result


@dataclass
class LlvmFunctionType(Type):
    result_type: Type
    argument_types: List[Type]

    def dump(self, indent: int = 0) -> str:
        result = dump_or_value(self.result_type) + ''
        if self.argument_types:
            result += ' (%s)' % ', '.join(
                dump_or_value(arg, indent) for arg in self.argument_types)
        return result


##############################################################################
# Attributes


# Default attribute implementation
class Attribute(Node):
    pass


@dataclass
class ArrayAttr(Attribute):
    value: List[Attribute]

    def dump(self, indent: int = 0) -> str:
        return '[%s]' % dump_or_value(self.value, indent)


@dataclass
class BoolAttr(Attribute):
    value: bool

    def dump(self, indent: int = 0) -> str:
        return dump_or_value(self.value, indent)


@dataclass
class DictionaryAttr(Attribute):
    value: List["AttributeEntry"]

    def dump(self, indent: int = 0) -> str:
        return '{%s}' % dump_or_value(self.value, indent)


@dataclass
class ElementsAttr(Attribute):
    pass


@dataclass
class DenseElementsAttr(ElementsAttr):
    attribute: Attribute
    type: Union[TensorType, VectorType]

    def dump(self, indent: int = 0) -> str:
        return 'dense<%s> : %s' % (self.attribute.dump(indent),
                                   self.type.dump(indent))


@dataclass
class OpaqueElementsAttr(ElementsAttr):
    dialect: str
    attribute: Attribute
    type: Union[TensorType, VectorType]

    def dump(self, indent: int = 0) -> str:
        return 'opaque<%s, %s> : %s' % (self.dialect,
                                        dump_or_value(self.attribute, indent),
                                        self.type.dump(indent))


@dataclass
class SparseElementsAttr(ElementsAttr):
    indices: List[List[int]]
    values: List[Any]
    type: Type

    def dump(self, indent: int = 0) -> str:
        return 'sparse<%s, %s> : %s' % (dump_or_value(self.indices, indent),
                                        dump_or_value(self.values, indent),
                                        self.type.dump(indent))


@dataclass
class PrimitiveAttribute(Attribute):
    value: Any
    type: Type

    def dump(self, indent: int = 0) -> str:
        return dump_or_value(self.value, indent) + (
            ': %s' % self.type.dump(indent) if self.type is not None else '')


class FloatAttr(PrimitiveAttribute):
    pass


class IntegerAttr(PrimitiveAttribute):
    pass


class StringAttr(PrimitiveAttribute):
    pass


@dataclass
class IntSetAttr(Attribute):
    value: "AffineMap"

    def dump(self, indent: int = 0) -> str:
        return dump_or_value(self.value, indent)


@dataclass
class TypeAttr(Attribute):
    value: type

    def dump(self, indent: int = 0) -> str:
        return dump_or_value(self.value, indent)


@dataclass
class SymbolRefAttr(Attribute):
    path: List[SymbolRefId]

    def dump(self, indent: int = 0) -> str:
        return '::'.join(dump_or_value(p, indent) for p in self.path)


@dataclass
class UnitAttr(Attribute):
    def dump(self, indent: int = 0) -> str:
        return 'unit'


# Attribute entries
@dataclass
class AttributeEntry(Node):
    name: str
    value: Optional[Attribute]

    def dump(self, indent: int = 0) -> str:
        if self.value:
            return '%s = %s' % (dump_or_value(self.name, indent),
                                dump_or_value(self.value, indent))
        return dump_or_value(self.name, indent)


@dataclass
class DialectAttributeEntry(Node):
    dialect: str
    name: str
    value: Optional[Attribute] = None

    def dump(self, indent: int = 0) -> str:
        if self.value:
            return '%s.%s = %s' % (dump_or_value(self.dialect, indent),
                                   dump_or_value(self.name, indent),
                                   dump_or_value(self.value, indent))
        return '%s.%s' % (dump_or_value(self.dialect, indent),
                          dump_or_value(self.name, indent))


@dataclass
class AttributeDict(Node):
    values: List[AttributeEntry]

    def dump(self, indent: int = 0) -> str:
        return '{%s}' % ', '.join(
            dump_or_value(v, indent) for v in self.values)


##############################################################################
# Operations

@dataclass
class OpResult(Node):
    value: SsaId
    count: Optional[int] = None

    def dump(self, indent: int = 0) -> str:
        return self.value.dump(indent) + (
            (':' + dump_or_value(self.count, indent)) if self.count else '')


@dataclass
class Operation(Node):
    result_list: List[OpResult]
    op: "Op"
    location: Optional["Location"] = None

    def dump(self, indent: int = 0) -> str:
        result = indent * '  '
        if self.result_list:
            result += '%s = ' % (', '.join(
                dump_or_value(r, indent) for r in self.result_list))
        result += dump_or_value(self.op, indent)
        if self.location:
            result += ' ' + self.location.dump(indent)
        return result


class Op(Node):
    pass


@dataclass
class GenericOperation(Op):
    name: str
    args: Optional[List[SsaId]]
    successors: Optional[List[BlockId]]
    regions: Optional[List["Region"]]
    attributes: Optional[AttributeDict]
    type: List[Type]

    def dump(self, indent: int = 0) -> str:
        result = '%s' % self.name
        result += '('

        if self.args:
            result += ', '.join(dump_or_value(arg, indent) for arg in self.args)

        result += ')'
        if self.successors:
            result += '[' + dump_or_value(self.successors, indent) + ']'
        if self.regions:
            result += ' ( ' + ', '.join(r.dump(indent) for r in self.regions) + ')'
        if self.attributes:
            result += ' ' + dump_or_value(self.attributes, indent)
        if isinstance(self.type, list):
            result += ' : ' + ', '.join(
                dump_or_value(t, indent) for t in self.type)
        else:
            result += ' : ' + dump_or_value(self.type, indent)
        return result


@dataclass
class CustomOperation(Op):
    namespace: str
    name: str
    args: List[SsaId]
    type: List[Type]

    def dump(self, indent: int = 0) -> str:
        result = '%s.%s' % (self.namespace, self.name)
        if self.args:
            result += ' %s' % ', '.join(
                dump_or_value(arg, indent) for arg in self.args)
        if isinstance(self.type, list):
            result += ' : ' + ', '.join(
                dump_or_value(t, indent) for t in self.type)
        else:
            result += ' : ' + dump_or_value(self.type, indent)

        return result


class Location(Node):
    pass


@dataclass
class StrLocation(Node):
    value: str

    def dump(self, indent: int = 0) -> str:
        return 'loc(%s)' % dump_or_value(self.value, indent)


@dataclass
class FileLineColLoc(Location):
    file: str
    line: int
    col: int

    def dump(self, indent: int = 0) -> str:
        return 'loc(%s:%d:%d)' % (self.file, self.line, self.col)


##############################################################################
# Modules, functions, and blocks

class ModuleType(Node):
    pass

@dataclass
class Module(ModuleType):
    name: Optional[str]
    attributes: Optional[AttributeDict]
    region: "Region"
    location: Optional[Location] = None

    def dump(self, indent=0) -> str:
        result = 'module '
        if self.name:
            result += '%s ' % self.name.dump(indent)
        if self.attributes:
            result += ' attributes ' + dump_or_value(self.attributes, indent)

        result += self.region.dump(indent)
        if self.location:
            result += ' ' + self.location.dump(indent)
        return result


@dataclass
class GenericModule(ModuleType):
    name: str
    args: List["NamedArgument"]
    region: "Region"
    attributes: Optional[AttributeDict]
    type: List[Type]
    location: Optional[Location] = None

    def dump(self, indent=0) -> str:
        result = dump_or_value(self.name, indent)
        result += '('
        if self.args:
            result += ' %s' % ', '.join(
                dump_or_value(arg, indent) for arg in self.args)
        result += ')'
        result += ' ( '
        result += self.region.dump(indent)
        result += ')'
        if self.attributes:
            result += ' ' + dump_or_value(self.attributes, indent)
        result += ' : ' + self.type.dump(indent)
        if self.location:
            result += ' ' + self.location.dump(indent)
        return result


@dataclass
class Function(Node):
    name: SymbolRefId
    args: Optional[List["NamedArgument"]]
    result_types: Optional[List[Type]]
    attributes: Optional[Union[Attribute, AttrAlias]]
    region: Optional["Region"]
    location: Optional[Location] = None

    def dump(self, indent=0) -> str:
        result = 'func'
        result += ' %s' % self.name.dump(indent)
        if self.args:
            result += '(%s)' % ', '.join(
                dump_or_value(arg, indent) for arg in self.args)
        if self.result_types:
            if not isinstance(self.result_types, list):
                result += ' -> ' + dump_or_value(self.result_types, indent)
            else:
                result += ' -> (%s)' % ', '.join(
                    dump_or_value(res, indent) for res in self.result_types)
        if self.attributes:
            result += ' attributes ' + dump_or_value(self.attributes, indent)

        result += ' %s' % (self.region.dump(indent) if self.region else '{\n%s}' %
                           (indent * '  '))
        if self.location:
            result += ' ' + self.location.dump(indent)
        return result


@dataclass
class Region(Node):
    body: List['Block']

    def dump(self, indent=0) -> str:
        return ('{\n' + '\n'.join(
            op.dump(indent + 1)
            for op in self.body) + '\n%s}' % (indent * '  '))


@dataclass
class Block(Node):
    label: Optional["BlockLabel"]
    body: List[Operation]

    def dump(self, indent=0) -> str:
        result = ''
        if self.label:
            result += indent * '  ' + self.label.dump(indent)
            indent += 1
        result += '\n'.join(
            stmt.dump(indent) for stmt in self.body)
        return result


@dataclass
class BlockLabel(Node):
    name: BlockId
    arg_ids: Optional[List[SsaId]]
    arg_types: Optional[List[Type]]

    def dump(self, indent: int = 0) -> str:
        result = dump_or_value(self.name, indent)
        if self.arg_ids:
            result += ' (%s)' % (', '.join(
                f'{dump_or_value(id_, indent)}: {dump_or_value(type_, indent)}'
                for id_, type_ in zip(self.arg_ids, self.arg_types)))
        result += ':\n'
        return result


@dataclass
class NamedArgument(Node):
    name: SsaId
    type: Type
    attributes: Optional[Union[AttributeDict, AttrAlias]] = None

    def dump(self, indent: int = 0) -> str:
        result = '%s: %s' % (dump_or_value(self.name, indent),
                             dump_or_value(self.type, indent))
        if self.attributes:
            result += ' %s' % dump_or_value(self.attributes, indent)
        return result


@dataclass
class MLIRFile(Node):
    definitions: List["Definition"]
    modules: List[Module]

    def dump(self, indent: int = 0) -> str:
        result = ''
        if self.definitions:
            result += '\n'.join(dump_or_value(defn, indent)
                                for defn in self.definitions)

            result += '\n'

        if self.modules:
            result += dump_or_value(self.modules, indent)
        return result

    @property
    def default_module(self) -> Module:
        """
        If *self* contains exactly one module returns it, otherwise raises a
        :class:`ValueError`.
        """
        if len(self.modules) != 1:
            raise ValueError("Can access default_module iff the "
                             "MLIR file has exactly one module.")
        return self.modules[0]


##############################################################################
# Affine and semi-affine expressions


# Types of affine expressions
# Contents of single/multi-dimensional (semi-)affine expressions
class AffineExpr(Node):
    # TODO: Inserts a lot of "AffineParens" that  leading to a generated with high
    # SNR. Should solve this by strategically placing the parens so that the
    # precedence isn't violated.
    def __add__(self, other: Union["AffineExpr", int]):
        return AffineParens(AffineAdd(operand_a=self, operand_b=other))

    def __sub__(self, other: Union["AffineExpr", int]):
        return AffineParens(AffineSub(operand_a=self, operand_b=other))

    def __mul__(self, other: int):
        return AffineParens(AffineMul(operand_a=self, operand_b=other))

    def __neg__(self):
        return AffineParens(AffineNeg(operand=self))

    def __radd__(self, other: Union["AffineExpr", int]):
        return AffineParens(AffineAdd(operand_a=other, operand_b=self))

    def __rsub__(self, other: Union["AffineExpr", int]):
        return AffineParens(AffineSub(operand_a=other, operand_b=self))

    def __rmul__(self, other: int):
        return AffineParens(AffineMul(operand_a=other, operand_b=self))


class SemiAffineExpr(AffineExpr):
    def __floordiv__(self, other: int):
        return AffineParens(AffineFloorDiv(operand_a=self, operand_b=other))

    def __mod__(self, other: int):
        return AffineParens(AffineMod(operand_a=self, operand_b=other))


@dataclass
class MultiDimAffineExpr(Node):
    dims: List[AffineExpr]

    def dump(self, indent: int = 0) -> str:
        return '%s : (%s)' % (dump_or_value(self.dims_and_symbols, indent),
                              dump_or_value(self.constraints, indent))

    def dump(self, indent: int = 0) -> str:
        return '(%s)' % dump_or_value(self.dims, indent)


@dataclass
class MultiDimSemiAffineExpr(Node):
    dims: List[SemiAffineExpr]

    def dump(self, indent: int = 0) -> str:
        return '(%s)' % dump_or_value(self.dims, indent)


class AffineSsa(SsaId, AffineExpr):
    pass


@dataclass
class AffineDimOrSymbol(AffineExpr):
    value: str

    def dump(self, indent: int = 0) -> str:
        return self.value


@dataclass
class AffineUnaryOp(AffineExpr):
    operand: AffineExpr
    _op_: str = field(init=False, repr=False)

    def dump(self, indent: int = 0) -> str:
        return self._op_ % dump_or_value(self.operand, indent)


@dataclass
class AffineBinaryOp(AffineExpr):
    operand_a: Union[AffineExpr, int]
    operand_b: Union[AffineExpr, int]
    _op_: str = field(init=False, repr=False)

    def dump(self, indent: int = 0) -> str:
        return '%s %s %s' % (dump_or_value(self.operand_a, indent), self._op_,
                             dump_or_value(self.operand_b, indent))

class AffineNeg(AffineUnaryOp): _op_ = '-%s'
class AffineParens(AffineUnaryOp): _op_ = '(%s)'
class AffineExplicitSymbol(AffineUnaryOp): _op_ = 'symbol(%s)'

class AffineAdd(AffineBinaryOp): _op_ = '+'
class AffineSub(AffineBinaryOp): _op_ = '-'
class AffineMul(AffineBinaryOp): _op_ = '*'
class AffineFloorDiv(AffineBinaryOp): _op_ = 'floordiv'
class AffineCeilDiv(AffineBinaryOp): _op_ = 'ceildiv'
class AffineMod(AffineBinaryOp): _op_ = 'mod'


##############################################################################
# (semi-)Affine maps, and integer sets

@dataclass
class DimAndSymbolList(Node):
    dims: List[str]
    symbols: Optional[List[str]]

    def dump(self, indent: int = 0) -> str:
        if self.symbols:
            return '(%s)[%s]' % (dump_or_value(self.dims, indent),
                                 dump_or_value(self.symbols, indent))
        return '(%s)' % dump_or_value(self.dims, indent)


@dataclass
class AffineConstraint(Node):
    expr: AffineExpr


class AffineConstraintGreaterEqual(AffineConstraint):
    def dump(self, indent: int = 0) -> str:
        return '%s >= 0' % dump_or_value(self.expr, indent)


class AffineConstraintEqual(AffineConstraint):
    def dump(self, indent: int = 0) -> str:
        return '%s == 0' % dump_or_value(self.expr, indent)


@dataclass
class AffineMap(Node):
    dims_and_symbols: DimAndSymbolList
    map: MultiDimAffineExpr

    def dump(self, indent: int = 0) -> str:
        return 'affine_map<%s -> %s>' % (dump_or_value(self.dims_and_symbols, indent),
                             dump_or_value(self.map, indent))


@dataclass
class SemiAffineMap(Node):
    dims_and_symbols: DimAndSymbolList
    map: MultiDimSemiAffineExpr

    def dump(self, indent: int = 0) -> str:
        return '%s -> %s' % (dump_or_value(self.dims_and_symbols, indent),
                             dump_or_value(self.map, indent))


@dataclass
class IntSet(Node):
    dims_and_symbols: DimAndSymbolList
    constraints: Optional[List[AffineConstraint]]

    def dump(self, indent: int = 0) -> str:
        return '%s : (%s)' % (dump_or_value(self.dims_and_symbols, indent),
                              dump_or_value(self.constraints, indent) if self.constraints else '')


##############################################################################
# Top-level definitions


@dataclass
class Definition(Node):
    name: Identifier
    value: Any

    def dump(self, indent: int = 0) -> str:
        return (indent * '  ' + dump_or_value(self.name, indent) + ' = ' +
                dump_or_value(self.value, indent))


class TypeAliasDef(Definition):
    def dump(self, indent: int = 0) -> str:
        return (indent * '  ' + dump_or_value(self.name, indent) + ' = type ' +
                dump_or_value(self.value, indent))


class AttrAliasDef(Definition):
    pass


class AffineMapDef(Definition):
    pass


class SemiAffineMapDef(Definition):
    pass


class IntSetDef(Definition):
    pass


##############################################################################
# Helpers


def _dump_ast_or_value(value: Any, python=True, indent: int = 0) -> str:
    """ Helper function to dump the AST node type or a reconstructible
        node value.
        :param python: Use Python syntax for output.
    """
    if python and hasattr(value, 'dump_ast'):
        return value.dump_ast()
    if not python and hasattr(value, 'dump'):
        return value.dump(indent=indent)

    # Literals
    if not python and isinstance(value, bool):
        return 'true' if value else 'false'
    if python and isinstance(value, str):
        return '"%s"' % value

    # Primitive types
    if isinstance(value, list):
        if not python:
            return ', '.join(_dump_ast_or_value(v, python) for v in value)
        return '[%s]' % ', '.join(_dump_ast_or_value(v, python) for v in value)
    if isinstance(value, tuple):
        return '(%s%s)' % (', '.join(
            _dump_ast_or_value(v, python)
            for v in value), ', ' if python else '')
    if isinstance(value, dict):
        sep = ': ' if python else ' = '
        return '{%s}' % ', '.join(
            '%s%s%s' %
            (_dump_ast_or_value(k, python), sep, _dump_ast_or_value(v, python))
            for k, v in value.items())
    return str(value)


def dump_or_value(value: Any, indent: int = 0) -> str:
    """ Helper function to dump the MLIR value or a reconstructible
        node value. """
    return _dump_ast_or_value(value, python=False, indent=indent)
