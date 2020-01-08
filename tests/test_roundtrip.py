""" Tests pyMLIR in a parse->dump->parse round-trip. """

from mlir import parse_string


def test_toy_roundtrip():
    """ Create MLIR code without extra whitespace and check that it can parse
        and dump the same way. """
    code = '''module {
  func @toy_func(%arg0: tensor<2x3xf64>) -> tensor<3x2xf64> {
    %0 = "toy.transpose"(%arg0) {inplace = true} : (tensor<2x3xf64>) -> tensor<3x2xf64>
    return %0 : tensor<3x2xf64>
  }
}'''

    module = parse_string(code)
    dump = module.dump()
    assert dump == code


if __name__ == '__main__':
    test_toy_roundtrip()
