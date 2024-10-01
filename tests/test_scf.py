import mlir

# All source strings taken from examples in https://mlir.llvm.org/docs/Dialects/SCFDialect/


def assert_roundtrip_equivalence(source):
    assert source == mlir.parse_string(source).dump()


def test_scf_for():
    assert_roundtrip_equivalence("""module {
  func.func @reduce(%buffer: memref<1024xf32>, %lb: index, %ub: index, %step: index, %sum_0: f32) -> (f32) {
    %sum = scf.for %iv = %lb to %ub step %step iter_args ( %sum_iter = %sum_0 ) -> ( f32 ) {
      %t = load %buffer [ %iv ] : memref<1024xf32>
      %sum_next = arith.addf %sum_iter, %t : f32
      scf.yield %sum_next : f32
    }
    return %sum : f32
  }
}""")


def test_scf_if():
    assert_roundtrip_equivalence("""module {
  func.func @example(%A: f32, %B: f32, %C: f32, %D: f32) {
    %x, %y = scf.if %b -> ( f32, f32 ) {
      scf.yield %A, %B : f32, f32
    } else {
      scf.yield %C, %D : f32, f32
    }
    return
  }
}""")


def test_scf_while():
    assert_roundtrip_equivalence("""module {
  func.func @example(%A: f32, %B: f32, %C: f32, %D: f32) {
    %res = scf.while ( %arg1 = %init1 ) : (f32) -> f32 {
      %condition = func.call @evaluate_condition ( %arg1 ) : (f32) -> i1
      scf.condition ( %condition ) %arg1 : f32
    } do {
      ^bb0 (%arg2: f32):
        %next = func.call @payload ( %arg2 ) : (f32) -> f32
        scf.yield %next : f32
    }
  }
}""")


if __name__ == '__main__':
    test_scf_for()
    test_scf_if()
    test_scf_while()
