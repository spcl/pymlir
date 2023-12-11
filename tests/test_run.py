__copyright__ = "Copyright (C) 2020 Kaushik Kulkarni"

__license__ = """
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import numpy as np
import sys
import mlir.run as mlirrun
import pytest
from pytools.prefork import ExecError


def is_mlir_opt_present():
    try:
        mlirrun.get_mlir_opt_version()
        return True
    except ExecError:
        return False


@pytest.mark.skipif(not is_mlir_opt_present(), reason="mlir-opt not found")
def test_add():
    source = """
    #identity = affine_map<(i,j) -> (i,j)>
    #attrs = {
      indexing_maps = [#identity, #identity, #identity],
      iterator_types = ["parallel", "parallel"]
    }
    func @example(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>) {
      linalg.generic #attrs ins(%A, %B: memref<?x?xf64>, memref<?x?xf64>) outs(%C: memref<?x?xf64>) {
      ^bb0(%a: f64, %b: f64, %c: f64):
        %d = addf %a, %b : f64
        linalg.yield %d : f64
      }
      return
    }"""

    source = mlirrun.mlir_opt(source, ["-convert-linalg-to-loops",
                                       "-convert-scf-to-std"])
    a = np.random.rand(10, 10)
    b = np.random.rand(10, 10)
    c = np.empty_like(a)

    mlirrun.call_function(source, "example", [a, b, c])

    np.testing.assert_allclose(c, a+b)


@pytest.mark.skipif(not is_mlir_opt_present(), reason="mlir-opt not found")
def test_axpy():
    source = """
func @saxpy(%a : f32, %x : memref<?xf32>, %y : memref<?xf32>) {
  %c0 = constant 0: index
  %n = dim %x, %c0 : memref<?xf32>

  affine.for %i = 0 to %n {
    %xi = affine.load %x[%i] : memref<?xf32>
    %axi =  mulf %a, %xi : f32
    %yi = affine.load %y[%i] : memref<?xf32>
    %axpyi = addf %yi, %axi : f32
    affine.store %axpyi, %y[%i] : memref<?xf32>
  }
  return
}"""

    source = mlirrun.mlir_opt(source, ["-lower-affine",
                                       "-convert-scf-to-std"])
    alpha = np.float32(np.random.rand())
    x_in = np.random.rand(10).astype(np.float32)
    y_in = np.random.rand(10).astype(np.float32)
    y_out = y_in.copy()

    mlirrun.call_function(source, "saxpy", [alpha, x_in, y_out])

    np.testing.assert_allclose(y_out, alpha*x_in+y_in)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
