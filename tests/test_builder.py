import sys
from mlir import parse_string
from mlir.builder import IRBuilder
from mlir.builder import Reads, Writes, Isa
from mlir.dialects.affine import AffineLoadOp
from mlir.dialects.standard import AddfOperation


def test_saxpy_builder():
    builder = IRBuilder()
    F64 = builder.F64
    Mref1D = builder.MemRefType(shape=(None, ), dtype=F64)

    mlirfile = builder.make_mlir_file()
    module = mlirfile.default_module

    with builder.goto_block(builder.make_block(module.region)):
        saxpy_fn = builder.function("saxpy")

    block = builder.make_block(saxpy_fn.region)
    builder.position_at_entry(block)

    a, x, y = builder.add_function_args(saxpy_fn, [F64, Mref1D, Mref1D])
    c0 = builder.index_constant(0)
    n = builder.dim(x, c0, builder.INDEX)

    f = builder.affine.for_(0, n)
    i = f.index

    with builder.goto_block(builder.make_block(f.region)):
        axi = builder.mulf(builder.affine.load(x, i, Mref1D), a, F64)
        axpyi = builder.addf(builder.affine.load(y, i, Mref1D), axi, F64)
        builder.affine.store(axpyi, y, i, Mref1D)

    builder.ret()

    print(mlirfile.dump())


def test_query():
    block = parse_string("""
func @saxpy(%a : f64, %x : memref<?xf64>, %y : memref<?xf64>) {
%c0 = constant 0 : index
%n = dim %x, %c0 : memref<?xf64>

affine.for %i = 0 to %n {
  %xi = affine.load %x[%i+1] : memref<?xf64>
  %axi =  mulf %a, %xi : f64
  %yi = affine.load %y[%i] : memref<?xf64>
  %axpyi = addf %yi, %axi : f64
  affine.store %axpyi, %y[%i] : memref<?xf64>
}
return
}""").default_module.region.body[0].body[0].op.region.body[0]
    for_block = block.body[2].op.region.body[0]

    c0 = block.body[0].result_list[0].value

    def query(expr):
        return next((op
                   for op in block.body + for_block.body
                   if expr(op)))

    assert query(Writes("%c0")).dump() == "%c0 = constant 0 : index"
    assert (query(Reads("%y") & Isa(AffineLoadOp)).dump()
            == "%yi = affine.load %y [ %i ] : memref<?xf64>")

    assert query(Reads(c0)).dump() == "%n = dim %x , %c0 : memref<?xf64>"


def test_build_with_queries():
    builder = IRBuilder()
    F64 = builder.F64

    mlirfile = builder.make_mlir_file()
    module = mlirfile.default_module

    with builder.goto_block(builder.make_block(module.region)):
        fn = builder.function("unorderly_adds")

    a0, a1, b0, b1, c0, c1 = builder.add_function_args(fn, [F64]*6)

    fnbody = builder.make_block(fn.region)
    builder.position_at_entry(fnbody)

    def index(expr):
        return next((i
                   for i, op in enumerate(fnbody.body)
                   if expr(op)))

    builder.addf(a0, a1, F64)

    with builder.goto_before(Reads(a0) & Reads(a1)):
        builder.addf(b0, b1, F64)

    with builder.goto_after(Reads(b0) & Isa(AddfOperation)):
        builder.addf(c0, c1, F64)

    builder.ret()

    assert index(Reads(b0)) == 0
    assert index(Reads(c0)) == 1
    assert index(Reads(a0)) == 2


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
