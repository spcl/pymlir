""" Tests pyMLIR on different syntactic edge-cases. """

from mlir import parse_string


def test_attributes():
    code = '''
module {
  func @myfunc(%tensor: tensor<256x?xf64>) -> tensor<*xf64> {
    %t_tensor = "with_attributes"(%tensor) { inplace = true, abc = 123, bla = unit, hello_world = "hey", value=@this::@is::@hierarchical, somelist = ["of", "values"], last = {butnot = "least", dictionaries = 0xabc} } : (tensor<2x3xf64>) -> tuple<vector<3xi33>,tensor<2x3xf64>> 
    return %t_tensor : tensor<3x2xf64>
  }
}
    '''

    module = parse_string(code)
    print(module.pretty())

def test_memrefs():
    code = '''
module {
  func @myfunc() {
        %a, %b = "tensor_replicator"(%tensor, %tensor) : (memref<?xbf16, 2>, 
          memref<?xf32, offset: 5, strides: [6, 7]>,
          memref<*xf32, 8>)
  }
}
'''
    module = parse_string(code)
    print(module.pretty())


def test_trailing_loc():
    code = '''
    module {
      func @myfunc() {
        %c:2 = addf %a, %b : f32 loc("test_syntax.py":36:59)
      }
    } loc("hi.mlir":30:1)
    '''
    module = parse_string(code)
    print(module.pretty())


def test_modules():
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
    module = parse_string(code)
    print(module.pretty())

def test_functions():
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
    module = parse_string(code)
    print(module.pretty())


if __name__ == '__main__':
    test_attributes()
    test_memrefs()
    test_trailing_loc()
    test_modules()
    test_functions()
