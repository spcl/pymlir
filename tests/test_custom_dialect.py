""" Test that creates and uses a custom dialect. """

from typing import List, Union, Optional
from mlir import parse_string
from mlir.astnodes import Node, dump_or_value, SsaId, StringLiteral, TensorType, MemRefType, Dimension
from mlir.dialect import Dialect, DialectOp, DialectType
from mlir.dialects.func import func
from dataclasses import dataclass


##############################################################################
# Dialect Types

@dataclass
class RaggedTensorType(DialectType):
    """
    AST node class for the example "toy" dialect representing a ragged tensor.
    """
    implementation: "ToyImplementation"
    dims: List[Dimension]
    type: Union[TensorType, MemRefType]

    _syntax_ = 'toy.ragged < {implementation.toy_impl_list} , {dims.dimension_list_ranked} {type.tensor_memref_element_type} >'

    # Custom MLIR serialization implementation
    def dump(self, indent: int = 0) -> str:
        return '!toy.ragged<%s, %sx%s>' % (
            dump_or_value(self.implementation, indent),
            'x'.join(dump_or_value(d, indent) for d in self.dims),
            dump_or_value(self.type, indent)
        )


class ToyImplementation(Node):
    """ Base "toy" implementation AST node. Corresponds to a "+"-separated list
        of sparse tensor types.
    """
    _fields_ = ['values']

    def __init__(self, values: List[str]):
        self.values = values

    def dump(self, indent: int = 0) -> str:
        return '+'.join(dump_or_value(v, indent) for v in self.values)


##############################################################################
# Dialect Operations

@dataclass
class DensifyOp(DialectOp):
    """ AST node for an operation with an optional value. """
    arg: SsaId
    type: TensorType
    pad: Optional[Union[StringLiteral, float, int, bool]] = None
    _syntax_ = ['toy.densify {arg.ssa_id} : {type.tensor_type}',
                'toy.densify {arg.ssa_id} , {pad.constant_literal} : {type.tensor_type}']


##############################################################################
# Dialect

my_dialect = Dialect('toy', ops=[DensifyOp], types=[RaggedTensorType],
                     preamble='''
// Exclamation mark in Lark means that string tokens will be preserved upon parsing
!toy_impl_type : "coo" | "csr" | "csc" | "ell"
toy_impl_list   : toy_impl_type ("+" toy_impl_type)*
                     ''',
                     transformers=dict(
                         toy_impl_list=ToyImplementation,
                         # Will convert every instance to its contents
                         toy_impl_type=lambda v: v[0]
                     ))


##############################################################################
# Tests

def test_custom_dialect():
    code = '''module {
  func.func @toy_test(%ragged: !toy.ragged<coo+csr, 32x14xf64>) -> tensor<32x14xf64> {
    %t_tensor = toy.densify %ragged : tensor<32x14xf64>
    return %t_tensor : tensor<32x14xf64>
  }
}'''
    m = parse_string(code, dialects=[my_dialect])
    dump = m.pretty()
    print(dump)

    # Test for round-trip
    assert dump == code


if __name__ == '__main__':
    test_custom_dialect()
