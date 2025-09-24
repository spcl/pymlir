from .affine import affine as affine_dialect
from .cf import cf as cf_dialect
from .math import math as math_dialect
from .tensor import tensor as tensor_dialect
from .arith import arith as arith_dialect
from .scf import scf as scf_dialect
from .linalg import linalg
from .func import func as func_dialect
from .memref import memref as memref_dialect


STANDARD_DIALECTS = [affine_dialect, cf_dialect, math_dialect, tensor_dialect, arith_dialect, scf_dialect, linalg, func_dialect, memref_dialect]
