""" Tests pyMLIR on different syntactic edge-cases. """

from mlir import Parser
from mlir.dialects.func import func 
import pytest
from typing import Optional

@pytest.fixture
def parser(parser: Optional[Parser] = None) -> Parser:
    return parser if parser is not None else Parser()

def test_attributes(parser):
    code = '''
module {
  func.func @myfunc(%tensor: tensor<256x?xf64>) -> tensor<*xf64> {
    %t_tensor = "with_attributes"(%tensor) { inplace = true, abc = -123, bla = unit, hello_world = "hey", value=@this::@is::@hierarchical, somelist = ["of", "values"], last = {butnot = "least", dictionaries = 0xabc} } : (tensor<2x3xf64>) -> tuple<vector<3xi33>,tensor<2x3xf64>> 
    return %t_tensor : tensor<3x2xf64>
  }
  func.func @toy_func(%arg0: tensor<2x3xf64>) -> tensor<3x2xf64> {
    %0:2 = "toy.split"(%arg0) : (tensor<2x3xf64>) -> (tensor<3x2xf64>, f32)
    return %0#50 : tensor<3x2xf64>
  }
}
    '''
    parser = parser or Parser()
    module = parser.parse(code)
    print(module.pretty())


def test_memrefs(parser):
    code = '''
module {
  func.func @myfunc() {
        %a, %b = "tensor_replicator"(%tensor, %tensor) : (memref<?xbf16, 2>, 
          memref<?xf32, offset: 5, strides: [6, 7]>,
          memref<*xf32, 8>)
  }
}
    '''
    parser = parser or Parser()
    module = parser.parse(code)
    print(module.pretty())


def test_trailing_loc(parser):
    code = '''
    module {
      func.func @myfunc() {
        %c:2 = addf %a, %b : f32 loc("test_syntax.py":36:59)
      }
    } loc("hi.mlir":30:1)
    '''
    parser = parser or Parser()
    module = parser.parse(code)
    print(module.pretty())


def test_modules(parser):
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


def test_functions(parser):
    code = '''
    module {
      func.func @myfunc_a() {
        %c:2 = addf %a, %b : f32
      }
      func.func @myfunc_b() {
        %d:2 = addf %a, %b : f64
        ^e:
        %f:2 = addf %d, %d : f64
      }
    }'''
    parser = parser or Parser()
    module = parser.parse(code)
    print(module.pretty())


def test_toplevel_function(parser):
    code = '''
    func.func @toy_func(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
      %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
      return %t_tensor : tensor<3x2xf64>
    }'''

    parser = parser or Parser()
    module = parser.parse(code)
    print(module.pretty())


def test_toplevel_functions(parser):
    code = '''
    func.func @toy_func(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
      %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
      return %t_tensor : tensor<3x2xf64>
    }
    func.func @toy_func(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
      %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
      return %t_tensor : tensor<3x2xf64>
    }'''

    parser = parser or Parser()
    module = parser.parse(code)
    print(module.pretty())


def test_affine(parser):
    code = '''
func.func @empty() {
  affine.for %i = 0 to 10 {
  } {some_attr = true}
  %0 = affine.min affine_map<(d0)[s0] -> (1000, d0 + 512, s0)> (%arg0)[%arg1]
}
func.func @valid_symbols(%arg0: index, %arg1: index, %arg2: index) {
  %c0 = constant 1 : index
  %c1 = constant 0 : index
  %b = alloc()[%N] : memref<4x4xf32, (d0, d1)[s0] -> (d0, d0 + d1 + s0 floordiv 2)>
  %0 = alloc(%arg0, %arg1) : memref<?x?xf32>
  affine.for %arg3 = %arg1 to %arg2 step 768 {
    %13 = dim %0, %c1 : memref<?x?xf32>
    affine.for %arg4 = 0 to %13 step 264 {
      %18 = dim %0, %c0 : memref<?x?xf32>
      %20 = subview %0[%c0, %c0][%18,%arg4][%c1,%c1] : memref<?x?xf32>
                          to memref<?x?xf32, (d0, d1)[s0, s1, s2] -> (d0 * s1 + d1 * s2 + s0)>
      %24 = dim %20, %c0 : memref<?x?xf32, (d0, d1)[s0, s1, s2] -> (d0 * s1 + d1 * s2 + s0)>
      affine.for %arg5 = 0 to %24 step 768 {
        "foo"() : () -> ()
      }
    }
  }
  return
}
    '''
    parser = parser or Parser()
    module = parser.parse(code)
    print(module.pretty())


def test_definitions(parser):
    code = '''
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<() -> (10)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1 + d2 + 5)>
#map5 = affine_map<(d0, d1, d2) -> (d0 + d1, d2)>
#map6 = affine_map<(d0, d1)[s0] -> (d0, d1 + s0 + 7)>
#map7 = affine_map<(d0, d1)[s0] -> (d0 + s0, d1)>
#map8 = affine_map<(d0, d1) -> (d0 + d1 + 11)>
#map9 = affine_map<(d0, d1)[s0] -> (d0, (d1 + s0) mod 9 + 7)>
#map10 = affine_map<(d0, d1)[s0] -> ((d0 + s0) floordiv 3, d1)>
#samap0 = (d0)[s0] -> (d0 floordiv (s0 + 1))
#samap1 = (d0)[s0] -> (d0 floordiv s0)
#samap2 = (d0, d1)[s0, s1] -> (d0*s0 + d1*s1)
#set0 = (d0) : (1 == 0)
#set1 = (d0, d1)[s0] : ()
#set2 = (d0, d1)[s0, s1] : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + s1 - 1 >= 0)
#set3 = (d0, d1, d2) : (d0 - d2 * 4 == 0, d0 + d1 * 8 - 9 >= 0, -d0 - d1 * 8 + 11 >= 0)
#set4 = (d0, d1, d2, d3, d4, d5) : (d0 * 1089234 + d1 * 203472 + 82342 >= 0, d0 * -55 + d1 * 24 + d2 * 238 - d3 * 234 - 9743 >= 0, d0 * -5445 - d1 * 284 + d2 * 23 + d3 * 34 - 5943 >= 0, d0 * -5445 + d1 * 284 + d2 * 238 - d3 * 34 >= 0, d0 * 445 + d1 * 284 + d2 * 238 + d3 * 39 >= 0, d0 * -545 + d1 * 214 + d2 * 218 - d3 * 94 >= 0, d0 * 44 - d1 * 184 - d2 * 231 + d3 * 14 >= 0, d0 * -45 + d1 * 284 + d2 * 138 - d3 * 39 >= 0, d0 * 154 - d1 * 84 + d2 * 238 - d3 * 34 >= 0, d0 * 54 - d1 * 284 - d2 * 223 + d3 * 384 >= 0, d0 * -55 + d1 * 284 + d2 * 23 + d3 * 34 >= 0, d0 * 54 - d1 * 84 + d2 * 28 - d3 * 34 >= 0, d0 * 54 - d1 * 24 - d2 * 23 + d3 * 34 >= 0, d0 * -55 + d1 * 24 + d2 * 23 + d3 * 4 >= 0, d0 * 15 - d1 * 84 + d2 * 238 - d3 * 3 >= 0, d0 * 5 - d1 * 24 - d2 * 223 + d3 * 84 >= 0, d0 * -5 + d1 * 284 + d2 * 23 - d3 * 4 >= 0, d0 * 14 + d2 * 4 + 7234 >= 0, d0 * -174 - d2 * 534 + 9834 >= 0, d0 * 194 - d2 * 954 + 9234 >= 0, d0 * 47 - d2 * 534 + 9734 >= 0, d0 * -194 - d2 * 934 + 984 >= 0, d0 * -947 - d2 * 953 + 234 >= 0, d0 * 184 - d2 * 884 + 884 >= 0, d0 * -174 + d2 * 834 + 234 >= 0, d0 * 844 + d2 * 634 + 9874 >= 0, d2 * -797 - d3 * 79 + 257 >= 0, d0 * 2039 + d2 * 793 - d3 * 99 - d4 * 24 + d5 * 234 >= 0, d2 * 78 - d5 * 788 + 257 >= 0, d3 - (d5 + d0 * 97) floordiv 423 >= 0, ((d0 + (d3 mod 5) floordiv 2342) * 234) mod 2309 + (d0 + d3 * 2038) floordiv 208 >= 0, ((((d0 + d3 * 2300) * 239) floordiv 2342) mod 2309) mod 239423 == 0, d0 + d3 mod 2642 + (((((d3 + d0 * 2) mod 1247) mod 2038) mod 2390) mod 2039) floordiv 55 >= 0)
#matmul_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
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


def test_generic_dialect_std(parser):
    code = '''
"module"() ( {
  "func.func"() ( {
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
    %0 = "std.addi"(%arg1, %arg0) : (i32, i32) -> i32
    "std.return"(%0) : (i32) -> ()
  }) {sym_name = "mlir_entry", type = (i32, i32) -> i32} : () -> ()
}) : () -> ()
    '''
    parser = parser or Parser()
    module = parser.parse(code)
    print(module.pretty())

def test_generic_dialect_std_cond_br(parser):
    code = '''
"module"() ( {
"func.func"() ( {
^bb0(%arg0: i32):  // no predecessors
    %c1_i32 = "std.constant"() {value = 1 : i32} : () -> i32
    %0 = "std.cmpi"(%arg0, %c1_i32) {predicate = 3 : i64} : (i32, i32) -> i1
    "std.cond_br"(%0)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (i1) -> ()
^bb1:  // pred: ^bb0
    "std.return"(%c1_i32) : (i32) -> ()
^bb2:  // pred: ^bb0
    "std.return"(%c1_i32) : (i32) -> ()
}) {sym_name = "mlir_entry", type = (i32) -> i32} : () -> ()
}) : () -> ()
    '''
    parser = parser or Parser()
    module = parser.parse(code)
    print(module.pretty())

def test_generic_dialect_llvm(parser):
    code = '''
"module"() ( {
  "llvm.func"() ( {
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
    %0 = "llvm.add"(%arg1, %arg0) : (i32, i32) -> i32
    "llvm.return"(%0) : (i32) -> ()
  }) {linkage = 10 : i64, sym_name = "mlir_entry", type = !llvm.func<i32 (i32, i32)>} : () -> ()
}) : () -> ()
    '''
    parser = parser or Parser()
    module = parser.parse(code)
    print(module.pretty())


def test_generic_dialect_generic_op(parser):
    code = '''
"module"() ( {
  "func.func"() ( {
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
    %0 = "generic_op_with_region"(%arg0, %arg1) ( {
      %1 = "std.addi"(%arg1, %arg0) : (i32, i32) -> i32
      "std.return"(%1) : (i32) -> ()
    }) : (i32, i32) -> i32
    %2 = "generic_op_with_regions"(%0, %arg0) ( {
      %3 = "std.subi"(%0, %arg0) : (i32, i32) -> i32
      "std.return"(%3) : (i32) -> ()
    }, {
      %4 = "std.addi"(%3, %arg0) : (i32, i32) -> i32
      "std.return"(%4) : (i32) -> ()
    }) : (i32, i32) -> i32
    %5 = "generic_op_with_region_and_attr"(%2, %arg0) ( {
      %6 = "std.subi"(%2, %arg0) : (i32, i32) -> i32
      "std.return"(%6) : (i32) -> ()
    }) {attr = "string attribute"} : (i32, i32) -> i32
    %7 = "generic_op_with_region_and_successor"(%5, %arg0)[^bb1] ( {
      %8 = "std.addi"(%5, %arg0) : (i32, i32) -> i32
      "std.br"(%8)[^bb1] : (i32) -> ()
    }) {attr = "string attribute"} : (i32, i32) -> i32
  ^bb1(%ret: i32):
    "std.return"(%ret) : (i32) -> ()
  }) {sym_name = "mlir_entry", type = (i32, i32) -> i32} : () -> ()
}) : () -> ()
    '''
    parser = parser or Parser()
    module = parser.parse(code)
    print(module.pretty())


def test_integer_sign(parser):
    code = '''
func.func @integer_test(%a: si16, %b: ui32, %c: i7) {
  return
}
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
    test_affine(p)
    test_definitions(p)
    test_generic_dialect_std(p)
    test_generic_dialect_std_cond_br(p)
    test_generic_dialect_llvm(p)
    test_generic_dialect_generic_op(p)
    test_integer_sign(p)
