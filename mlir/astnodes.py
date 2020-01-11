""" Classes containing MLIR AST node types, fields, and conversion back to
    MLIR. """

from enum import Enum, auto
from typing import Any, List, Union
from lark import Token


class Node(object):
    """ Base MLIR AST node object. """

    # Static field defining which fields should be used in this node
    _fields_: List[str] = []

    def __init__(self, node: Token = None, **fields):
        # Set each field separately
        if node is not None and isinstance(node, list):
            for fname, fval in zip(self._fields_, node):
                setattr(self, fname, fval)

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
        return (type(self).__name__ + '(' + ', '.join(
            f + '=' + _dump_ast_or_value(getattr(self, f))
            for f in self._fields_) + ')')

    def dump(self) -> str:
        """ Dumps the AST node and its children in MLIR format.
            :return: String representing the AST in MLIR.
        """
        return '<UNIMPLEMENTED>'

    def __repr__(self):
        return (type(self).__name__ + '(' + ', '.join(
            f + '=' + str(getattr(self, f)) for f in self._fields_) + ')')

    def pretty(self):
        return self.dump()
        # result = self.dump_ast()
        # lines = ['']
        # indent = 0
        # for char in result:
        #     if char == '[':
        #         indent += 1
        #     if char == ']':
        #         indent -= 1
        #     if char != '\n':
        #         lines[-1] += char
        #     if char in '[\n':
        #         lines.append(indent * '  ')
        #
        # return '\n'.join(lines)


class StringLiteral(object):
    def __init__(self, value: str):
        self.value = value

    def __str__(self):
        return '"%s"' % self.value

    def __repr__(self):
        return '"%s"' % self.value


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
    _fields_ = ['value', 'index']
    _prefix_ = '%'

    def __init__(self, node: Token = None, **fields):
        self.value = node[0]
        self.index = node[1] if len(node) > 1 else None
        super().__init__(None, **fields)

    def dump(self) -> str:
        if self.index:
            return self._prefix_ + self.value + ("#%s" % self.index)
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


class Dimension(Type):
    _fields_ = ['value']

    def __init__(self, node: Token = None, **fields):
        self.value = None
        try:
            if isinstance(node[0], int):
                self.value = node[0]
        except (IndexError, TypeError):
            pass  # In case of an unknown size

        super().__init__(None, **fields)

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

    def __init__(self, node: Token = None, **fields):
        super().__init__(node, **fields)
        if 'type' not in fields:
            self.type = FloatTypeEnum[node[0]]

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

    def __init__(self, node: Token = None, **fields):
        self.types = node
        super().__init__(None, **fields)

    def dump(self) -> str:
        return 'tuple<%s>' % (', '.join(t.dump() for t in self.types))


class VectorType(Type):
    _fields_ = ['dimensions', 'element_type']

    def dump(self) -> str:
        return 'vector<%s>' % ('x'.join(
            _dump_or_value(t)
            for t in self.dimensions) + 'x' + self.element_type.dump())


class RankedTensorType(Type):
    _fields_ = ['dimensions', 'element_type']

    def dump(self) -> str:
        return 'tensor<%s>' % ('x'.join(t.dump() for t in self.dimensions) +
                               'x' + self.element_type.dump())


class UnrankedTensorType(Type):
    _fields_ = ['element_type']

    def __init__(self, node: Token = None, **fields):
        # Ignore unranked dimension list
        super().__init__(node[1:], **fields)

    def dump(self) -> str:
        return 'tensor<*x%s>' % self.element_type.dump()


class RankedMemRefType(Type):
    _fields_ = ['dimensions', 'element_type', 'layout', 'space']

    def __init__(self, node: Token = None, **fields):
        self.dimensions = node[0]
        self.element_type = node[1]
        self.layout = None
        self.space = None
        if len(node) > 2:
            if node[2].data == 'memory_space':
                self.space = node[2].children[0]
            elif node[2].data == 'layout_specification':
                self.layout = node[2].children[0]

        super().__init__(None, **fields)

    def dump(self) -> str:
        result = 'memref<%s' % ('x'.join(t.dump() for t in self.dimensions) +
                                'x' + self.element_type.dump())
        if self.layout is not None:
            result += ', ' + self.layout.dump()
        if self.space is not None:
            result += ', ' + _dump_or_value(self.space)

        return result + '>'


class UnrankedMemRefType(Type):
    _fields_ = ['element_type', 'space']

    def __init__(self, node: Token = None, **fields):
        self.element_type = node[0]
        self.space = None
        if len(node) > 1:
            self.space = node[1].children[0]

        super().__init__(None, **fields)

    def dump(self) -> str:
        result = 'memref<%s' % ('*x' + self.element_type.dump())
        if self.space is not None:
            result += ', ' + _dump_or_value(self.space)

        return result + '>'


class OpaqueDialectType(Type):
    _fields_ = ['dialect', 'contents']

    def dump(self) -> str:
        return '!%s<"%s">' % (self.dialect, self.contents)


class PrettyDialectType(Type):
    _fields_ = ['dialect', 'type', 'body']

    def dump(self) -> str:
        return '!%s.%s<%s>' % (
            self.dialect, self.type, ', '.join(_dump_or_value(item)
                                               for item in self.body))


class FunctionType(Type):
    _fields_ = ['argument_types', 'result_types']

    def dump(self) -> str:
        result = '(%s)' % ', '.join(
            _dump_or_value(arg) for arg in self.argument_types)
        result += ' -> '
        if not self.result_types:
            result += '()'
        elif len(self.result_types) == 1:
            result += _dump_or_value(self.result_types[0])
        else:
            result += '(%s)' % ', '.join(
                _dump_or_value(res) for res in self.result_types)
        return result


class StridedLayout(Node):
    _fields_ = ['offset', 'strides']

    def dump(self) -> str:
        return 'offset: %s, strides: %s' % (_dump_or_value(self.offset),
                                            _dump_or_value(self.strides))


##############################################################################
# Attributes


# Attribute entries
class AttributeEntry(Node):
    _fields_ = ['name', 'value']

    def __init__(self, node: Token = None, **fields):
        self.name = node[0]
        self.value = node[1] if len(node) > 1 else None
        super().__init__(None, **fields)

    def dump(self) -> str:
        if self.value:
            return '%s = %s' % (_dump_or_value(self.name),
                                _dump_or_value(self.value))
        return _dump_or_value(self.name)


class DialectAttributeEntry(Node):
    _fields_ = ['dialect', 'name', 'value']

    def __init__(self, node: Token = None, **fields):
        self.dialect = node[0]
        self.name = node[1]
        self.value = node[2] if len(node) > 2 else None
        super().__init__(None, **fields)

    def dump(self) -> str:
        if self.value:
            return '%s.%s = %s' % (_dump_or_value(self.dialect),
                                   _dump_or_value(self.name),
                                   _dump_or_value(self.value))
        return '%s.%s' % (_dump_or_value(self.dialect),
                          _dump_or_value(self.name))


class AttributeDict(Node):
    _fields_ = ['values']

    def __init__(self, node: Token = None, **fields):
        self.values = node
        super().__init__(None, **fields)

    def dump(self) -> str:
        return '{%s}' % ', '.join(_dump_or_value(v) for v in self.values)


# Default attribute implementation
class Attribute(Node):
    _fields_ = ['value']

    def dump(self) -> str:
        return _dump_or_value(self.value)


class ArrayAttr(Attribute):
    def __init__(self, node: Token = None, **fields):
        self.value = node
        super().__init__(None, **fields)

    def dump(self) -> str:
        return '[%s]' % _dump_or_value(self.value)


class BoolAttr(Attribute):
    pass


class DictionaryAttr(Attribute):
    def __init__(self, node: Token = None, **fields):
        self.value = node
        super().__init__(None, **fields)

    def dump(self) -> str:
        return '{%s}' % _dump_or_value(self.value)


class ElementsAttr(Attribute):
    pass


class DenseElementsAttr(ElementsAttr):
    _fields_ = ['attribute', 'type']

    def dump(self) -> str:
        return 'dense<%s> : %s' % (self.attribute.dump(), self.type.dump())


class OpaqueElementsAttr(ElementsAttr):
    _fields_ = ['dialect', 'attribute', 'type']

    def dump(self) -> str:
        return 'opaque<%s, %s> : %s' % (
            self.dialect, _dump_or_value(self.attribute), self.type.dump())


class SparseElementsAttr(ElementsAttr):
    _fields_ = ['indices', 'values', 'type']

    def dump(self) -> str:
        return 'sparse<%s, %s> : %s' % (_dump_or_value(
            self.indices), _dump_or_value(self.values), self.type.dump())


class PrimitiveAttribute(Attribute):
    _fields_ = ['value', 'type']

    def __init__(self, node: Token = None, **fields):
        self.value = node[0]
        if len(node) > 1:
            self.type = node[1]
        else:
            self.type = None

        super().__init__(None, **fields)

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

    def __init__(self, node: Token = None, **fields):
        self.path = node
        super().__init__(None, **fields)

    def dump(self) -> str:
        return '::'.join(_dump_or_value(p) for p in self.path)


class UnitAttr(Attribute):
    _fields_ = []

    def dump(self) -> str:
        return 'unit'


##############################################################################
# Operations


class OpResult(Node):
    _fields_ = ['value', 'count']

    def __init__(self, node: Token = None, **fields):
        self.value = node[0]
        if len(node) > 1:
            self.count = node[1]
        else:
            self.count = None
        super().__init__(None, **fields)

    def dump(self) -> str:
        return self.value.dump() + (
            (':' + _dump_or_value(self.count)) if self.count else '')


class Operation(Node):
    _fields_ = ['result_list', 'op', 'location']

    def __init__(self, node: Token = None, **fields):
        index = 0
        if isinstance(node[0], list):
            self.result_list = node[index]
            index += 1
        else:
            self.result_list = []
        self.op = node[index]
        index += 1
        if len(node) > index:
            self.location = node[2]
        else:
            self.location = None

        super().__init__(None, **fields)

    def dump(self) -> str:
        result = ''
        if self.result_list:
            result += '%s = ' % (', '.join(
                _dump_or_value(r) for r in self.result_list))
        result += _dump_or_value(self.op)
        if self.location:
            result += ' ' + self.location.dump()
        return result


class Op(Node):
    pass


class GenericOperation(Op):
    _fields_ = ['name', 'args', 'attributes', 'type']

    def __init__(self, node: Token = None, **fields):
        index = 0
        self.name = node[index]
        index += 1
        if len(node) > index and isinstance(node[index], list):
            self.args = node[index]
            index += 1
        else:
            self.args = []
        if len(node) > index and isinstance(node[index], AttributeDict):
            self.attributes = node[index]
            index += 1
        else:
            self.attributes = None
        if len(node) > index:
            self.type = node[index]
        else:
            self.type = None

        super().__init__(None, **fields)

    def dump(self) -> str:
        result = '%s' % self.name
        result += '(%s)' % ', '.join(_dump_or_value(arg) for arg in self.args)
        if self.attributes:
            result += ' ' + _dump_or_value(self.attributes)
        if isinstance(self.type, list):
            result += ' : ' + ', '.join(_dump_or_value(t) for t in self.type)
        else:
            result += ' : ' + _dump_or_value(self.type)
        return result


class CustomOperation(Op):
    _fields_ = ['namespace', 'name', 'args', 'type']

    def __init__(self, node: Token = None, **fields):
        self.namespace = node[0]
        self.name = node[1]
        if len(node) == 4:
            self.args = node[2]
            self.type = node[3]
        else:
            self.args = None
            self.type = node[2]

        super().__init__(None, **fields)

    def dump(self) -> str:
        result = '%s.%s' % (self.namespace, self.name)
        if self.args:
            result += ' %s' % ', '.join(
                _dump_or_value(arg) for arg in self.args)
        if isinstance(self.type, list):
            result += ' : ' + ', '.join(_dump_or_value(t) for t in self.type)
        else:
            result += ' : ' + _dump_or_value(self.type)

        return result


class Location(Node):
    _fields_ = ['value']

    def dump(self) -> str:
        return 'loc(%s)' % _dump_or_value(self.value)


class FileLineColLoc(Location):
    _fields_ = ['file', 'line', 'col']

    def dump(self) -> str:
        return 'loc(%s:%d:%d)' % (self.file, self.line, self.col)


##############################################################################
# Modules, functions, and blocks


class Module(Node):
    _fields_ = ['name', 'attributes', 'body', 'location']

    def __init__(self, node: Union[Token, Node] = None, **fields):
        index = 0
        if isinstance(node, Node):
            self.name = None
            self.attributes = None
            self.body = [node]
            self.location = None
        else:
            if len(node) > index and isinstance(node[index], SymbolRefId):
                self.name = node[index]
                index += 1
            else:
                self.name = None
            if len(node) > index and isinstance(node[index], AttributeDict):
                self.attributes = node[index]
                index += 1
            else:
                self.attributes = None
            self.body = node[index].children
            index += 1
            if len(node) > index:
                self.location = node[index]
            else:
                self.location = None

        super().__init__(None, **fields)

    def dump(self, indent=0) -> str:
        result = indent * '  ' + 'module'
        if self.name:
            result += ' %s' % self.name.dump()
        if self.attributes:
            result += ' attributes ' + _dump_or_value(self.attributes)

        result += ' {\n'
        result += '\n'.join(block.dump(indent + 1) for block in self.body)
        result += '\n' + indent * '  ' + '}'
        if self.location:
            result += ' ' + self.location.dump()
        return result


class Function(Node):
    _fields_ = [
        'name', 'args', 'result_types', 'attributes', 'body', 'location'
    ]

    def __init__(self, node: Token = None, **fields):
        signature = node[0].children
        # Parse signature
        index = 0
        self.name = signature[index]
        index += 1
        if len(signature) > index and signature[index].data == 'argument_list':
            self.args = signature[index].children
            index += 1
        else:
            self.args = []
        if (len(signature) > index
                and signature[index].data == 'function_result_list'):
            self.result_types = signature[index].children
            index += 1
        else:
            self.result_types = []

        # Parse rest of function
        index = 1
        if len(node) > index and isinstance(node[index], AttributeDict):
            self.attributes = node[index]
            index += 1
        else:
            self.attributes = None
        if len(node) > index and isinstance(node[index], list):
            self.body = node[index]
            index += 1
        else:
            self.body = []
        if len(node) > index:
            self.location = node[index]
        else:
            self.location = None

        super().__init__(None, **fields)

    def dump(self, indent=0) -> str:
        result = indent * '  ' + 'func'
        result += ' %s' % self.name.dump()
        result += '(%s)' % ', '.join(_dump_or_value(arg) for arg in self.args)
        if self.result_types:
            if len(self.result_types) == 1:
                result += ' -> ' + _dump_or_value(self.result_types[0])
            else:
                result += ' -> (%s)' % ', '.join(
                    _dump_or_value(res) for res in self.result_types)
        if self.attributes:
            result += ' attributes ' + _dump_or_value(self.attributes)

        result += ' {\n'
        result += '\n'.join(
            block.dump(indent + 1) for region in self.body for block in region)
        result += '\n' + indent * '  ' + '}'
        if self.location:
            result += ' ' + self.location.dump()
        return result


class Block(Node):
    _fields_ = ['label', 'body']

    def __init__(self, node: Token = None, **fields):
        index = 0
        if len(node) > index and isinstance(node[index], BlockLabel):
            self.label = node[index]
            index += 1
        else:
            self.label = None
        if len(node) > index:
            self.body = node[index:]
        else:
            self.body = []

        super().__init__(None, **fields)

    def dump(self, indent=0) -> str:
        result = ''
        if self.label:
            result += indent * '  ' + self.label.dump()
        result += '\n'.join(indent * '  ' + stmt.dump() for stmt in self.body)
        return result


class BlockLabel(Node):
    _fields_ = ['name', 'args']

    def __init__(self, node: Token = None, **fields):
        self.name = node[0]
        if len(node) > 1:
            self.args = node[1]
        else:
            self.args = []

        super().__init__(None, **fields)

    def dump(self) -> str:
        result = _dump_or_value(self.name)
        if self.args:
            result += ' (%s)' % (', '.join(
                _dump_or_value(arg) for arg in self.args))
        result += ':\n'
        return result


class NamedArgument(Node):
    _fields_ = ['name', 'type', 'attributes']

    def __init__(self, node: Token = None, **fields):
        self.name = node[0]
        self.type = node[1]
        self.attributes = node[2] if len(node) > 2 else None
        super().__init__(None, **fields)

    def dump(self) -> str:
        result = '%s: %s' % (_dump_or_value(self.name),
                             _dump_or_value(self.type))
        if self.attributes:
            result += ' %s' % _dump_or_value(self.attributes)
        return result


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


def _dump_or_value(value: Any) -> str:
    """ Helper function to dump the MLIR value or a reconstructible
        node value. """
    return _dump_ast_or_value(value, python=False)
