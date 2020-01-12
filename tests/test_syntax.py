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


def test_definitions(parser: Optional[Parser] = None):
    code = '''
#map0 = (d0, d1) -> (d0, d1)
#map1 = (d0) -> (d0)
#map2 = () -> (0)
#map3 = () -> (10)
#map4 = (d0, d1, d2) -> (d0, d1 + d2 + 5)
#map5 = (d0, d1, d2) -> (d0 + d1, d2)
#map6 = (d0, d1)[s0] -> (d0, d1 + s0 + 7)
#map7 = (d0, d1)[s0] -> (d0 + s0, d1)
#map8 = (d0, d1) -> (d0 + d1 + 11)
#map9 = (d0, d1)[s0] -> (d0, (d1 + s0) mod 9 + 7)
#map10 = (d0, d1)[s0] -> ((d0 + s0) floordiv 3, d1)
//#samap0 = (d0)[s0] -> (d0 floordiv (s0 + 1))
#samap1 = (d0)[s0] -> (d0 floordiv s0)
#samap2 = (d0, d1)[s0, s1] -> (d0*s0 + d1*s1)
#set0 = (d0) : (1 == 0)
#set1 = (d0, d1)[s0] : ()
#matmul_accesses = [
  (m, n, k) -> (m, k),
  (m, n, k) -> (k, n),
  (m, n, k) -> (m, n)
]
#matmul_trait = {
  args_in = 2,
  args_out = 1,
  iterator_types = ["parallel", "parallel", "reduction"],
  indexing_maps = #matmul_accesses,
  library_call = "external_outerproduct_matmul"
}

!vector_type_A = type vector<4xf32>
!vector_type_B = type vector<4xf32>
!vector_type_C = type vector<4x4xf32>

!matrix_type_A = type memref<?x?x!vector_type_A>
!matrix_type_B = type memref<?x?x!vector_type_B>
!matrix_type_C = type memref<?x?x!vector_type_C>
    '''
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
    test_definitions(p)
