""" Classes containing MLIR AST node types, fields, and conversion back to
    MLIR. """

from enum import Enum, auto
from typing import Any, List
from lark import Token


class Node(object):
    """ Base MLIR AST node object. """

    # Static field defining which fields should be used in this node
    _fields_: List[str] = []

    def __init__(self, node: Token = None, **fields):
        # Set the defined fields
        for k, v in fields.items():
            setattr(self, k, v)

    def dump_ast(self) -> str:
        """ Dumps the AST node and its fields in raw AST format. For example:
            Module(name="example", body=[])

            :note: Due to the way objects are constructed, this format can be
                   parsed back by Python to the same AST.
            :return: String representing the AST in its raw format.
        """
        return (type(self).__name__ + '(' +
                ', '.join(f + '=' + _dump_ast_or_value(getattr(self, f))
                          for f in self._fields_) +
                ')')

    def dump(self) -> str:
        """ Dumps the AST node and its children in MLIR format.
            :return: String representing the AST in MLIR.
        """
        return '<UNIMPLEMENTED>'


##############################################################################
# Identifiers

class Identifier(Node):
    _fields_ = ['value']

    # Static field representing the prefix of this identifier. Used for ease
    # of MLIR output
    _prefix_: str = ''

    def dump(self) -> str:
        return self._prefix_ + self.value


class SsaId(Identifier):
    _prefix_ = '%'


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


class Dimension(Type):
    _fields_ = ['value']

    def dump(self) -> str:
        return str(self.value or '?')


class NoneType(Type):
    def dump(self) -> str:
        return 'none'


class FloatTypeEnum(Enum):
    f16 = auto()
    bf16 = auto()
    f32 = auto()
    f64 = auto()


class FloatType(Type):
    _fields_ = ['type']

    def dump(self) -> str:
        return self.type.name


class IndexType(Type):
    def dump(self) -> str:
        return 'index'


class IntegerType(Type):
    _fields_ = ['width']

    def dump(self) -> str:
        return 'i' + str(self.width)


class ComplexType(Type):
    _fields_ = ['type']

    def dump(self) -> str:
        return 'complex<%s>' % self.type.dump()


class TupleType(Type):
    _fields_ = ['types']

    def dump(self) -> str:
        return 'tuple<%s>' % (', '.join(t.dump() for t in self.types))


class VectorType(Type):
    _fields_ = ['dimensions', 'element_type']

    def dump(self) -> str:
        return 'vector<%s>' % ('x'.join(t.dump() for t in self.dimensions) +
                               'x' + self.element_type.dump())


class RankedTensorType(Type):
    _fields_ = ['dimensions', 'element_type']

    def dump(self) -> str:
        return 'tensor<%s>' % ('x'.join(t.dump() for t in self.dimensions) +
                               'x' + self.element_type.dump())


class UnrankedTensorType(Type):
    _fields_ = ['element_type']

    def dump(self) -> str:
        return 'tensor<*x%s>' % self.element_type.dump()


class RankedMemRefType(Type):
    _fields_ = ['dimensions', 'element_type', 'layout', 'space']

    def dump(self) -> str:
        result = 'memref<%s' % ('x'.join(t.dump() for t in self.dimensions) +
                                'x' + self.element_type.dump())
        if self.layout is not None:
            result += ', ' + self.layout.dump()
        if self.space is not None:
            result += ', ' + self.space.dump()

        return result + '>'


class UnrankedMemRefType(Type):
    _fields_ = ['element_type', 'space']

    def dump(self) -> str:
        result = 'memref<%s' % ('*x' + self.element_type.dump())
        if self.space is not None:
            result += ', ' + self.space.dump()

        return result + '>'


class OpaqueDialectType(Type):
    _fields_ = ['dialect', 'contents']

    def dump(self) -> str:
        return '!%s<"%s">' % (self.dialect, self.contents)


class PrettyDialectType(Type):
    _fields_ = ['dialect', 'type', 'body']

    def dump(self) -> str:
        return '!%s.%s<%s>' % (self.dialect, self.type,
                               _dump_or_value(self.body))


class FunctionType(Type):
    _fields_ = ['argument_types', 'result_types']

    def dump(self) -> str:
        return '%s -> %s' % (_dump_or_value(self.argument_types),
                             _dump_or_value(self.result_types))


##############################################################################
# Attributes

# Default attribute implementation
class Attribute(Node):
    _fields_ = ['value']

    def dump(self) -> str:
        return _dump_or_value(self.value)


class ArrayAttr(Attribute):
    _fields_ = ['values']

    def dump(self) -> str:
        return _dump_or_value(self.values)


class BoolAttr(Attribute):
    _fields_ = ['value']

    def dump(self) -> str:
        return _dump_or_value(self.value)


class DictionaryAttr(Attribute):
    _fields_ = ['values']

    def dump(self) -> str:
        return _dump_or_value(self.values)


class ElementsAttr(Attribute):
    pass


class DenseElementsAttr(ElementsAttr):
    _fields_ = ['attribute', 'type']

    def dump(self) -> str:
        return 'dense<%s> : %s' % (self.attribute.dump(), self.type.dump())


class OpaqueElementsAttr(ElementsAttr):
    _fields_ = ['dialect', 'attribute', 'type']

    def dump(self) -> str:
        return 'opaque<%s, %s> : %s' % (self.dialect,
                                        _dump_or_value(self.attribute),
                                        self.type.dump())


class SparseElementsAttr(ElementsAttr):
    _fields_ = ['indices', 'values', 'type']

    def dump(self) -> str:
        return 'sparse<%s, %s> : %s' % (_dump_or_value(self.indices),
                                        _dump_or_value(self.values),
                                        self.type.dump())


class PrimitiveAttribute(Attribute):
    _fields_ = ['value', 'type']

    def dump(self) -> str:
        return _dump_or_value(self.value) + (': %s' % self.type.dump()
                                             if self.type is not None else '')


class FloatAttr(PrimitiveAttribute):
    pass


class IntegerAttr(PrimitiveAttribute):
    pass


class StringAttr(PrimitiveAttribute):
    pass


class IntSetAttr(Attribute):
    pass  # Use default implementation


class TypeAttr(Attribute):
    pass  # Use default implementation


class SymbolRefAttr(Attribute):
    _fields_ = ['path']

    def dump(self) -> str:
        return '::'.join(_dump_or_value(p) for p in self.path)


class UnitAttr(Attribute):
    def dump(self) -> str:
        return 'unit'


##############################################################################
# Operations

class Operation(Node):
    _fields_ = ['result_list', 'op', 'args', 'attributes', 'type', 'location']

    def dump(self) -> str:
        result = ''
        if self.result_list:
            result += '%s =' % _dump_or_value(self.result_list)
        result += _dump_or_value(self.op)
        result += '(%s)' % ', '.join(_dump_or_value(arg) for arg in self.args)
        if self.attributes:
            result += _dump_or_value(self.attributes)
        result += ' : ' + self.type.dump()
        if self.location:
            result += ' ' + self.location.dump()

class Location(Node):
    _fields_ = ['value']

    def dump(self) -> str:
        return 'loc(%s)' % _dump_or_value(self.value)


class FileLineColLoc(Location):
    _fields_ = ['file', 'line', 'col']

    def dump(self) -> str:
        return 'loc("%s":%d:%d)' % (self.file, self.line, self.col)


##############################################################################
# Modules, functions, and blocks

class Module(Node):
    _fields_ = ['name', 'attributes', 'body', 'location']

    def dump(self, indent=0) -> str:
        result = indent*'  ' + 'module'
        if self.name:
            result += ' %s' % self.name.dump()
        if self.attributes:
            result += ' attributes ' + _dump_or_value(self.attributes)

        result += ' {\n'
        result += '\n'.join(block.dump(indent + 1) for block in self.body)
        result += '\n' + indent*'  ' + '}'
        if self.location:
            result += ' ' + self.location.dump()


class Function(Node):
    _fields_ = ['name', 'args', 'result_types', 'attributes', 'body',
                'location']

    def dump(self, indent=0) -> str:
        result = indent*'  ' + 'func'
        result += ' %s' % self.name.dump()
        result += '(%s)' % ', '.join(_dump_or_value(arg) for arg in self.args)
        if self.result_types:
            result += ' -> ' + _dump_or_value(self.result_types)
        if self.attributes:
            result += ' attributes ' + _dump_or_value(self.attributes)

        result += ' {\n'
        result += '\n'.join(block.dump(indent + 1) for block in self.body)
        result += '\n' + indent*'  ' + '}'
        if self.location:
            result += ' ' + self.location.dump()


class Block(Node):
    _fields_ = ['label', 'body']

    def dump(self, indent=0) -> str:
        result = ''
        if self.label:
            result += indent*'  ' + self.label.dump()
        result += '\n'.join(indent*'  ' + stmt.dump() for stmt in self.body)


##############################################################################
# (semi-)Affine expressions, maps, and integer sets
# TODO: Implement

class AffineExpr(Node):
    pass


class SemiAffineExpr(Node):
    pass


class MultiDimAffineExpr(Node):
    pass


class MultiDimSemiAffineExpr(Node):
    pass


class AffineConstraint(Node):
    pass


class AffineMap(Node):
    pass


class SemiAffineMap(Node):
    pass


class IntSet(Node):
    pass


##############################################################################
# Top-level definitions
# TODO: Implement

class Definition(Node):
    pass


class TypeAliasDef(Definition):
    pass


class AffineMapDef(Definition):
    pass


class SemiAffineMapDef(Definition):
    pass


class IntSetDef(Definition):
    pass


##############################################################################
# Helpers

def _dump_ast_or_value(value: Any, python=True) -> str:
    """ Helper function to dump the AST node type or a reconstructible
        node value.
        :param python: Use Python syntax for output.
    """
    if python and hasattr(value, 'dump_ast'):
        return value.dump_ast()
    if not python and hasattr(value, 'dump'):
        return value.dump()

    # Literals
    if not python and isinstance(value, bool):
        return 'true' if value else 'false'
    if python and isinstance(value, str):
        return '"%s"' % value

    # Primitive types
    if isinstance(value, list):
        return '[%s]' % ', '.join(_dump_ast_or_value(v, python) for v in value)
    if isinstance(value, tuple):
        return '(%s%s)' % (
            ', '.join(_dump_ast_or_value(v, python) for v in value),
            ', ' if python else '')
    if isinstance(value, dict):
        sep = ': ' if python else ' = '
        return '{%s}' % ', '.join(
            '%s%s%s' % (_dump_ast_or_value(k, python), sep,
                        _dump_ast_or_value(v, python)) for k, v in value.items())

    return str(value)


def _dump_or_value(value: Any) -> str:
    """ Helper function to dump the MLIR value or a reconstructible
        node value. """
    return _dump_ast_or_value(value, python=False)

