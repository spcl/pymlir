""" Implementation of the math (Mathematics) dialect. """

import inspect
import sys
from mlir.dialect import Dialect, DialectOp, is_op, UnaryOperation
import mlir.astnodes as mast
from dataclasses import dataclass
from typing import Optional, List, Tuple


# Unary Operations
class AbsfOperation(UnaryOperation): _opname_ = 'math.absf'
class CosOperation(UnaryOperation): _opname_ = 'math.cos'
class ExpOperation(UnaryOperation): _opname_ = 'math.exp'
class TanhOperation(UnaryOperation): _opname_ = 'math.tanh'
class CopysignOperation(UnaryOperation): _opname_ = 'math.copysign'

# Inspect current module to get all classes defined above
math = Dialect('math', ops=[m[1] for m in inspect.getmembers(
               sys.modules[__name__], lambda obj: is_op(obj, __name__))])
