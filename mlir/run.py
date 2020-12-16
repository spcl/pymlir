""" MLIR kernel invocation."""

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


import ctypes
import tempfile
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Any, Optional
from pytools import memoize_method
from pytools.prefork import call_capture_output, ExecError
from codepy.jit import compile_from_string
from codepy.toolchain import ToolchainGuessError, GCCToolchain
from codepy.toolchain import guess_toolchain as guess_toolchain_base


# {{{ Memref

def get_nd_memref_struct_type(n: int):
    nd_long = ctypes.c_long * n

    class NDMemrefStruct(ctypes.Structure):
        _fields_ = [("data", ctypes.c_void_p),
                    ("alignedData", ctypes.c_void_p),
                    ("offset", ctypes.c_long),
                    ("shape", nd_long),
                    ("strides", nd_long)]

    return NDMemrefStruct


@dataclass(init=True)
class Memref:
    data_ptr: int
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]

    @staticmethod
    def from_numpy(ary):
        """
        Create a :class:`Memref` from a :class:`numpy.ndarray`
        """
        shape = ary.shape
        strides = tuple(stride // ary.itemsize for stride in ary.strides)
        return Memref(ary.ctypes.data,
                      shape,
                      strides)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    @memoize_method
    def ctype(self):
        struct_cls = get_nd_memref_struct_type(self.ndim)

        typemap = dict(struct_cls._fields_)
        dataptr_cls = typemap["data"]
        shape_cls = typemap["shape"]
        strides_cls = typemap["strides"]

        return struct_cls(dataptr_cls(self.data_ptr),
                          dataptr_cls(self.data_ptr),
                          0,  # offset is alway zero for numpy arrays
                          shape_cls(*self.shape),
                          strides_cls(*self.strides))

    @property
    @memoize_method
    def pointer_ctype(self):
        return ctypes.pointer(self.ctype)

# }}}


# {{{ run kernels

def guess_toolchain():
    # copied from loopy/target/c/c_execution.py
    try:
        toolchain = guess_toolchain_base()
    except (ToolchainGuessError, ExecError):
        # missing compiler python was built with (likely, Conda)
        # use a default GCCToolchain
        # this is ugly, but I'm not sure there's a clean way to copy the
        # default args
        toolchain = GCCToolchain(
            cc="gcc",
            cflags="-std=c99 -O3 -fPIC".split(),
            ldflags=["-shared"],
            libraries=[],
            library_dirs=[],
            defines=[],
            undefines=[],
            source_suffix="c",
            so_ext=".so",
            o_ext=".o",
            include_dirs=[])

    return toolchain


def get_mlir_opt_version(mlir_opt="mlir-opt"):
    cmdline = [mlir_opt, "-version"]
    result, stdout, stderr = call_capture_output(cmdline)
    return stdout.decode()


def mlir_opt(source: str, options: List[str], mlir_opt="mlir-opt"):
    """
    Calls ``mlir-opt`` on *source* with *options* as additional arguments.

    :arg source: The code to be passed to mlir-opt.
    :arg options: An instance of :class:`list`.
    :return: Transformed *source* as emitted by ``mlir-opt``.
    """
    assert "-o" not in options
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir") as fp:
        fp.write(source)
        fp.file.flush()

        cmdline = [mlir_opt, fp.name] + options
        result, stdout, stderr = call_capture_output(cmdline)

    return stdout.decode()


def mlir_translate(source, options, mlir_translate="mlir-translate"):
    """
    Calls ``mlir-translate`` on *source* with *options* as additional arguments.

    :arg source: The code to be passed to mlir-translate.
    :arg options: An instance of :class:`list`.
    :return: Transformed *source* as emitted by ``mlir-translate``.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as fp:
        fp.write(source)
        fp.file.flush()
        cmdline = [mlir_translate, fp.name] + options
        result, stdout, stderr = call_capture_output(cmdline)

    return stdout.decode()


def mlir_to_llvmir(source, debug=False):
    """
    Converts MLIR *source* to LLVM IR. Invokes ``mlir-tranlate -mlir-to-llvmir``
    under the hood.
    """
    if debug:
        return mlir_translate(source, ["-mlir-to-llvmir", "-debugify-level=location+variables"])
    else:
        return mlir_translate(source, ["-mlir-to-llvmir"])


def llvmir_to_obj(source, llc="llc"):
    """
    Returns the compiled object code for the LLVM code *source*.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ll") as llfp:
        llfp.write(source)
        llfp.file.flush()
        with tempfile.NamedTemporaryFile(suffix=".o", mode="rb") as objfp:
            cmdline = [llc, llfp.name, "-o", objfp.name, "-filetype=obj"]
            result, stdout, stderr = call_capture_output(cmdline)

            obj_code = objfp.read()

            return obj_code


def preprocess_arg(arg):
    if isinstance(arg, Memref):
        return arg.pointer_ctype
    elif isinstance(arg, np.ndarray):
        return Memref.from_numpy(arg).pointer_ctype
    elif isinstance(arg, np.number):
        return arg
    else:
        raise NotImplementedError(f"Unknown type: {type(arg)}.")


def guess_argtypes(args):
    argtypes = []
    for arg in args:
        if isinstance(arg, Memref):
            argtypes.append(ctypes.c_void_p)
        elif isinstance(arg, np.ndarray):
            argtypes.append(ctypes.c_void_p)
        elif isinstance(arg, np.number):
            argtypes.append(np.ctypeslib.as_ctypes_type(arg.dtype))
        else:
            raise NotImplementedError(f"Unknown type: {type(arg)}.")

    return argtypes


def call_function(source: str, fn_name: str, args: List[Any],
              argtypes: Optional[List[ctypes._SimpleCData]] = None):
    """
    Calls the function *fn_name* in *source*.

    :arg source: The MLIR code whose function is to be called.
    :arg args: A list of args to be passed to the function. Each arg can have
        one of the following types:
        - :class:`numpy.ndarray`
        - :class:`numpy.number
        - :class:`Memref`
    :arg fn_name: Name of the function op which is the to be called
    """

    source = mlir_opt(source, ["-convert-std-to-llvm=emit-c-wrappers"])
    fn_name = f"_mlir_ciface_{fn_name}"

    if argtypes is None:
        argtypes = guess_argtypes(args)

    args = [preprocess_arg(arg) for arg in args]

    obj_code = llvmir_to_obj(mlir_to_llvmir(source))

    toolchain = guess_toolchain()

    _, mod_name, ext_file, recompiled = \
        compile_from_string(toolchain, fn_name, obj_code,
                ["module.o"],
                source_is_binary=True)

    f = ctypes.CDLL(ext_file)
    fn = getattr(f, fn_name)
    fn.argtypes = argtypes
    fn.restype = None
    fn(*args)

# }}}


# vim: fdm=marker
