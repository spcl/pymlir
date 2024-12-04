[![codecov](https://codecov.io/gh/spcl/pymlir/branch/master/graph/badge.svg)](https://codecov.io/gh/spcl/pymlir)


# pyMLIR: Python Interface for the Multi-Level Intermediate Representation

pyMLIR is a full Python interface to parse, process, output and run [MLIR](https://mlir.llvm.org/) files according to the
syntax described in the [MLIR documentation](https://github.com/llvm/llvm-project/tree/master/mlir/docs). pyMLIR 
supports the basic dialects and can be extended with other dialects. It uses [Lark](https://github.com/lark-parser/lark)
to parse the MLIR syntax, and mirrors the classes into Python classes. Custom dialects can also be implemented with a
Python string-format-like syntax, or via direct parsing.

Note that the tool *does not depend on LLVM or MLIR*. It can be installed and invoked directly from Python. 

## Instructions 

**How to install:** `pip install pymlir`

**Requirements:** Python 3.6 or newer, and the requirements in `setup.py` or `requirements.txt`. To manually install the
requirements, use `pip install -r requirements.txt`

**Problem parsing MLIR files?** Run the file through LLVM's `mlir-opt` as `mlir.run.mlir_opt(source, ["--mlir-print-op-generic"])` to
get the generic form of the IR (instructions on how to build/install MLIR can be found [here](https://mlir.llvm.org/getting_started/)):
```python
source = mlir.run.mlir_opt(source, ["--mlir-print-op-generic"])
```

**Found other problems parsing files?** Not all dialects and modes are supported. Feel free to send us an issue or
create a pull request! This is a community project and we welcome any contribution.

## Usage examples

### Parsing MLIR files into Python

```python
import mlir

# Read a file path, file handle (stream), or a string
ast1 = mlir.parse_path('/path/to/file.mlir')
ast2 = mlir.parse_file(open('/path/to/file.mlir', 'r'))
ast3 = mlir.parse_string('''
module {
  func.func @toy_func(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
    %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
    return %t_tensor : tensor<3x2xf64>
  }
}
''')
```

### Inspecting MLIR files in Python

MLIR files can be inspected by dumping their contents (which will print standard MLIR code), or by using the same tools
as you would with Python's [ast](https://docs.python.org/3/library/ast.html) module.

```python
import mlir

# Dump valid MLIR files
m = mlir.parse_path('/path/to/file.mlir')
print(m.dump())

print('---')

# Dump the AST directly
print(m.dump_ast())

print('---')

# Or visit each node type by implementing visitor functions
class MyVisitor(mlir.NodeVisitor):
    def visit_Function(self, node: mlir.astnodes.Function):
        print('Function detected:', node.name.value)
        
MyVisitor().visit(m)
```

### Transforming MLIR files

MLIR files can also be transformed with a Python-like 
[NodeTransformer](https://docs.python.org/3/library/ast.html#ast.NodeTransformer) object.

```python
import mlir

m = mlir.parse_path('/path/to/file.mlir')

# Simple node transformer that removes all operations with a result
class RemoveAllResultOps(mlir.NodeTransformer):
    def visit_Operation(self, node: mlir.astnodes.Operation):
        # There are one or more outputs, return None to remove from AST
        if len(node.result_list) > 0:
            return None
            
        # No outputs, no need to do anything
        return self.generic_visit(node)
        
m = RemoveAllResultOps().visit(m)

# Write back to file
with open('output.mlir', 'w') as fp:
    fp.write(m.dump())
```

### Using custom dialects

Custom dialects can be written and loaded as part of the pyMLIR parser. [See full tutorial here](doc/custom_dialect.rst).

```python
import mlir
from lark import UnexpectedCharacters
from .mydialect import dialect

# Try to parse as-is
try:
    m = mlir.parse_path('/path/to/matrixfile.mlir')
except UnexpectedCharacters:  # MyMatrix dialect not recognized
    pass
    
# Add dialect to the parser
m = mlir.parse_path('/path/to/matrixfile.mlir', 
                    dialects=[dialect])

# Print output back
print(m.dump_ast())
```

### MLIR from scratch with the builder API

pyMLIR has a Builder API that can create MLIR ASTs on the fly within Python code.

```python
import mlir.builder

builder = mlir.builder.IRBuilder()
mlirfile = builder.make_mlir_file()
module = mlirfile.default_module

with builder.goto_block(builder.make_block(module.region)):
    hello = builder.function("hello_world")
    block = builder.make_block(hello.region)
    builder.position_at_entry(block)

    x, y = builder.add_function_args(hello, [builder.F64, builder.F64], ['a', 'b'])

    adder = builder.addf(x, y, builder.F64)
    builder.func.ret([adder], [builder.F64])

print(mlirfile.dump())
```
prints:
```mlir
module {
  func.func @hello_world(%a: f64, %b: f64) {
    %_pymlir_ssa = addf %a , %b : f64
    return %_pymlir_ssa : f64
  }
}
```

See also [saxpy](tests/test_builder.py) for a full example that registers and uses a dialect in the builder.

### Built-in dialect implementations and more examples

All dialect implementations can be found in the [dialects](mlir/dialects) subfolder. Additional uses
of the library, including a custom dialect implementation, can be found in the [tests](tests)
subfolder.


### Call `mlir-opt` and invoke functions

Note that invoking MLIR functions depends on LLVM toolchain. The following binaries must be present in `$PATH`:
- `mlir-opt`
- `mlir-translate`
- `llc`

```python
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

assert (c == a+b).all()
```
