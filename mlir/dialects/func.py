
import inspect
import sys
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

import mlir.astnodes as mast
from mlir.dialect import Dialect, DialectOp, is_op

Literal = Union[mast.StringLiteral, float, int, bool]
SsaUse = Union[mast.SsaId, Literal]

@dataclass
class CallIndirectOperation(DialectOp):
    func: mast.SymbolRefId
    type: mast.FunctionType
    args: Optional[List[SsaUse]] = None
    _syntax_ = ['func.call_indirect {func.symbol_ref_id} () : {type.function_type}',
                'func.call_indirect {func.symbol_ref_id} ( {args.ssa_use_list} ) : {type.function_type}']


@dataclass
class CallOperation(DialectOp):
    func: mast.SymbolRefId
    type: mast.FunctionType
    args: Optional[List[SsaUse]] = None
    _syntax_ = ['func.call {func.symbol_ref_id} () : {type.function_type}',
                'func.call {func.symbol_ref_id} ( {args.ssa_use_list} ) : {type.function_type}']

@dataclass
class ConstantOperation(DialectOp):
    value: mast.SymbolRefId
    type: mast.Type
    _syntax_ = ['func.constant {value.symbol_ref_id} : {type.type}']

# Note: The 'func.func' operation is defined as 'function' in mlir.lark.

@dataclass
class ReturnOperation(DialectOp):
    values: Optional[List[SsaUse]] = None
    types: Optional[List[mast.Type]] = None
    _syntax_ = ['return',
                'return {values.ssa_use_list} : {types.type_list_no_parens}']

    def dump(self, indent: int = 0) -> str:
        output = 'return'
        if self.values:
            output += ' ' + ', '.join([v.dump(indent) for v in self.values])
        if self.types:
            output += ' : ' + ', '.join([t.dump(indent) for t in self.types])

        return output

    

# Inspect current module to get all classes defined above
func = Dialect('func', ops=[m[1] for m in inspect.getmembers(
    sys.modules[__name__], lambda obj: is_op(obj, __name__))])
