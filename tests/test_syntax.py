""" Tests pyMLIR on different syntactic edge-cases. """

from mlir import Parser
from typing import Optional


def test_attributes(parser: Optional[Parser] = None):
    code = '''
module {
  func @myfunc(%tensor: tensor<256x?xf64>) -> tensor<*xf64> {
    %t_tensor = "with_attributes"(%tensor) { inplace = true, abc = -123, bla = unit, hello_world = "hey", value=@this::@is::@hierarchical, somelist = ["of", "values"], last = {butnot = "least", dictionaries = 0xabc} } : (tensor<2x3xf64>) -> tuple<vector<3xi33>,tensor<2x3xf64>> 
    return %t_tensor : tensor<3x2xf64>
  }
  func @toy_func(%arg0: tensor<2x3xf64>) -> tensor<3x2xf64> {
    %0:2 = "toy.split"(%arg0) : (tensor<2x3xf64>) -> (tensor<3x2xf64>, f32)
    return %0#50 : tensor<3x2xf64>
  }
}
    '''
    parser = parser or Parser()
    module = parser.parse(code)
    print(module.pretty())


def test_memrefs(parser: Optional[Parser] = None):
    code = '''
module {
  func @myfunc() {
        %a, %b = "tensor_replicator"(%tensor, %tensor) : (memref<?xbf16, 2>, 
          memref<?xf32, offset: 5, strides: [6, 7]>,
          memref<*xf32, 8>)
  }
}
    '''
    parser = parser or Parser()
    module = parser.parse(code)
    print(module.pretty())


def test_trailing_loc(parser: Optional[Parser] = None):
    code = '''
    module {
      func @myfunc() {
        %c:2 = addf %a, %b : f32 loc("test_syntax.py":36:59)
      }
    } loc("hi.mlir":30:1)
    '''
    parser = parser or Parser()
    module = parser.parse(code)
    print(module.pretty())


def test_modules(parser: Optional[Parser] = None):
    code = '''
module {
  module {
  }
  module {
  }
  module attributes {foo.attr = true} {
  }
  module {
    %1 = "foo.result_op"() : () -> i32
  }
  module {
  }
  %0 = "op"() : () -> i32
  module @foo {
    module {
      module @bar attributes {foo.bar} {
      }
    }
  }
}'''
    parser = parser or Parser()
    module = parser.parse(code)
    print(module.pretty())


def test_functions(parser: Optional[Parser] = None):
    code = '''
    module {
      func @myfunc_a() {
        %c:2 = addf %a, %b : f32
      }
      func @myfunc_b() {
        %d:2 = addf %a, %b : f64
        ^e:
        %f:2 = addf %d, %d : f64
      }
    }'''
    parser = parser or Parser()
    module = parser.parse(code)
    print(module.pretty())


def test_toplevel_function(parser: Optional[Parser] = None):
    code = '''
    func @toy_func(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
      %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
      return %t_tensor : tensor<3x2xf64>
    }'''

    parser = parser or Parser()
    module = parser.parse(code)
    print(module.pretty())


def test_toplevel_functions(parser: Optional[Parser] = None):
    code = '''
    func @toy_func(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
      %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
      return %t_tensor : tensor<3x2xf64>
    }
    func @toy_func(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
      %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
      return %t_tensor : tensor<3x2xf64>
    }'''

    parser = parser or Parser()
    module = parser.parse(code)
    print(module.pretty())


if __name__ == '__main__':
    p = Parser()
    print("MLIR parser created")
    test_attributes(p)
    test_memrefs(p)
    test_trailing_loc(p)
    test_modules(p)
    test_functions(p)
    test_toplevel_function(p)
    test_toplevel_functions(p)
