""" MLIR IR Builder."""

import mlir.astnodes as mast
import mlir.dialects.standard as std
import mlir.dialects.affine as affine
from typing import Optional, Tuple, Union, List, Any
from contextlib import contextmanager
from mlir.builder.match import Reads, Writes, Isa, All, And, Or, Not  # noqa: F401
from mlir.builder.match import MatchExpressionBase
from mlir.builder.ung import UniqueNameGenerator
from dataclasses import dataclass


__doc__ = """
.. currentmodule:: mlir.builder.builder

.. autoclass:: IRBuilder
.. autoclass:: AffineBuilder
"""


class IRBuilder:
    """
    MLIR AST builder. Provides convenience methods for adding core dialect
    operations to a :class:`~mlir.astnodes.Block`.

    .. attribute:: block

        The block that the builder is operating on.

    .. attribute:: position

        An instance of :class:`int`, indicating the position where the next
        operation is to be added in the :attr:`block`.

    .. note::

        * The concepts here at not true to the implementation in
          llvm-project/mlir. It should be seen more of a convenience to emit MLIR
          modules.

        * This class shared design elements from :class:`llvmlite.ir.IRBuilder`,
          querying mechanism from :mod:`loopy`.

    *Registering a custom dialect builder*

    .. automethod:: register_dialect

    *Position/block manipulation*

    .. automethod:: position_at_entry
    .. automethod:: position_at_exit
    .. automethod:: goto_block
    .. automethod:: goto_entry_block
    .. automethod:: goto_before
    .. automethod:: goto_after

    *Types*

    :attr F16: f16 type
    :attr F32: f32 type
    :attr F64: f64 type
    :attr INT32: i32 type
    :attr INT64: i64 type
    :attr INDEX: index type

    .. automethod:: MemRefType

    *Standard dialect ops*

    .. automethod:: dim
    .. automethod:: addf
    .. automethod:: mulf
    .. automethod:: index_constant
    .. automethod:: float_constant
    """

    def __init__(self):
        self.block = None
        self.position = None

        self._dialects = {
            "affine": AffineBuilder(self),
            "std": self,  # std dialect ops can also be globally referenced
        }

    name_gen = UniqueNameGenerator(forced_prefix="_pymlir_")

    F16 = mast.FloatType(type=mast.FloatTypeEnum.f16)
    F32 = mast.FloatType(type=mast.FloatTypeEnum.f32)
    F64 = mast.FloatType(type=mast.FloatTypeEnum.f64)
    INT32 = mast.IntegerType(32)
    INT64 = mast.IntegerType(64)
    INDEX = mast.IndexType()

    def __getattr__(self, item: str) -> Any:
        if item in self._dialects:
            return self._dialects[item]

        return super().__getattr__(item)

    def register_dialect(self, name: str,
                         dialect_builder: "DialectBuilder",
                         overwrite: bool = False) -> None:
        if name in self._dialects and not overwrite:
            raise ValueError(f"'{name}' already registered as a dialect.")

        self._dialects[name] = dialect_builder

    @staticmethod
    def make_mlir_file(module: Optional[mast.Module] = None) -> mast.MLIRFile:
        """
        Returns an instance of :class:`mlir.astnodes.MLIRFile` for *module*.  If
        *module* is *None*, defaults it with an empty :class:`mlir.astnodes.Module`.
        """
        if module is None:
            module = mast.Module(None, None, mast.Region([]))
        return mast.MLIRFile([], [module])

    def module(self, name: Optional[str] = None) -> mast.Module:
        """
        Inserts a :class:`mlir.astnodes.Module` with name *name* into *block*.

        Returns the inserted module.
        """
        if name is None:
            name = None
        else:
            name = mast.SymbolRefId(name)

        op = mast.Module(name, None, mast.Region([]))
        self._insert_op_in_block([], op)
        return op

    def function(self, name: Optional[str] = None) -> mast.Function:
        """
        Inserts a :class:`mlir.astnodes.Function` with name *name* into *block*.

        Returns the inserted function.
        """
        if name is None:
            name = self.name_gen("fn")

        op = mast.Function(mast.SymbolRefId(value=name), [], [], None,
                           mast.Region([]))

        self._insert_op_in_block([], op)
        return op

    @classmethod
    def make_block(cls, region: mast.Region, name: Optional[str] = None
                   ) -> mast.Block:
        """
        Appends a :class:`mlir.astnodes.Block` with name *name* to the *region*.

        Returns the appended block.
        """
        if name is None:
            label = None
        else:
            label = mast.BlockLabel(name, [], [])

        block = mast.Block(label, [])
        region.body.append(block)
        return block

    @classmethod
    def add_function_args(cls, function: mast.Function, dtypes: List[mast.Type],
                          names: Optional[List[str]] = None,
                          positions: Optional[List[int]] = None):
        """
        Adds arguments to *function*.

        :arg dtypes: Types of the arguments to be added to the function.
        :arg names: Names of the arguments to be added to the function.
        :arg positions: Positions where the arguments are to be inserted.
        """
        if names is None:
            names = [cls.name_gen("fnarg") for _ in dtypes]

        if function.args is None:
            function.args = []

        if positions is None:
            positions = list(range(len(function.args),
                                   len(function.args) + len(dtypes)))

        args = []

        for name, dtype, pos in zip(names, dtypes, positions):
            arg = mast.SsaId(name)
            function.args.insert(pos, mast.NamedArgument(arg, dtype))
            args.append(arg)

        return args

    def MemRefType(self,
                   dtype: mast.Type,
                   shape: Optional[Tuple[Optional[int], ...]],
                   offset: Optional[int] = None,
                   strides: Optional[Tuple[Optional[int], ...]] = None
                   ) -> mast.MemRefType:
        """
        Returns an instance of :class:`mlir.astnodes.UnrankedMemRefType` if shape is
        *None*, else returns a :class:`mlir.astnodes.RankedMemRefType`.
        """
        if shape is None:
            assert strides is None
            return mast.UnrankedMemRefType(dtype)
        else:
            shape = tuple(mast.Dimension(dim) for dim in shape)
            if strides is None and offset is None:
                layout = None
            else:
                if offset is None:
                    offset = 0
                if strides is not None:
                    if len(shape) != len(strides):
                        raise ValueError("shapes and strides must be of tuples"
                                         " of same dimensionality.")
                layout = mast.StridedLayout(strides, offset)

            return mast.RankedMemRefType(shape, dtype, layout)

    def _insert_op_in_block(self,
                            op_results: List[Optional[Union[mast.SsaId, str]]],
                            op):
        new_op_results = []
        for op_result in op_results:
            if op_result is None:
                op_result = self.name_gen("ssa")

            if isinstance(op_result, str):
                result = mast.SsaId(op_result)

            new_op_results.append(result)

        if self.block is None:
            raise ValueError("Not within any block to append")

        self.block.body.insert(self.position,
                               mast.Operation(result_list=new_op_results,
                                             op=op))
        self.position += 1

        if len(new_op_results) == 1:
            return new_op_results[0]
        elif len(new_op_results) > 1:
            return new_op_results
        else:
            return

    # {{{ position/block manipulation

    def position_at_entry(self, block: mast.Block):
        """
        Starts building at *block*'s entry.
        """
        self.block = block
        self.position = 0

    def position_at_exit(self, block: mast.Block):
        """
        Starts building at *block*'s exit.
        """
        self.block = block
        self.position = len(block.body)

    @contextmanager
    def goto_block(self, block: mast.Block):
        """
        Context to start building at *block*'s exit.

        Example usage::

            with builder.goto_block(block):
                # starts building at *block*'s exit.
                z = builder.addf(x, y, F64)

            # goes back to building at the builder's earlier position
        """
        parent_block = self.block
        parent_position = self.position

        self.position_at_exit(block)
        yield

        self.block = parent_block
        self.position = parent_position

    @contextmanager
    def goto_entry_block(self, block: mast.Block):
        """
        Context to start building at *block*'s entry.

        Example usage::

            with builder.goto_block(block):
                # starts building at *block*'s entry.
                z = builder.addf(x, y, F64)

            # goes back to building at the builder's earlier position
        """
        parent_block = self.block
        parent_position = self.position

        self.position_at_entry(block)
        yield

        self.block = parent_block
        self.position = parent_position

    def position_before(self, query: MatchExpressionBase,
                        block: Optional[mast.Block] = None):
        """
        Positions the builder to the point just before *query* gets matched in
        *block*.

        :arg block: Block to query the operations in. Defaults to the builder's
            block.

        Example usage::

            builder.position_before(Reads("%c0") & Isa(AddfOperation))
            # starts building before operation of form "... = addf %c0, ..."
        """
        if block is not None:
            self.block = block

        try:
            self.position = next((i
                                  for i, op in enumerate(self.block.body)
                                  if query(op)))
        except StopIteration:
            raise ValueError(f"Did not find an operation matching '{query}'.")

    def position_after(self, query, block: Optional[mast.Block] = None):
        """
        Positions the builder to the point just after *query* gets matched in
        *block*.

        :arg block: Block to query the operations in. Defaults to the builder's
            block.

        Example usage::

            builder.position_after(Writes("%c0") & Isa(ConstantOperation))
            # starts building after operation of form "%c0 = constant ...: ..."
        """
        if block is not None:
            self.block = block

        try:
            self.position = next((i
                                  for i, op in zip(range(len(self.block.body)-1,
                                                         -1, -1),
                                                   reversed(self.block.body))
                                  if query(op))) + 1
        except StopIteration:
            raise ValueError(f"Did not find an operation matching '{query}'.")

    @contextmanager
    def goto_before(self, query: MatchExpressionBase,
                    block: Optional[mast.Block] = None):
        """
        Enters a context to build at the point just before *query* gets matched in
        *block*.

        :arg block: Block to query the operations in. Defaults to the builder's
            block.

        Example usage::

            with builder.goto_before(Reads("%c0") & Isa(AddfOperation)):
                # starts building before operation of form "... = addf %c0, ..."
                z = builder.mulf(x, y, F64)
            # goes back to building at the builder's earlier position
        """
        parent_block = self.block
        parent_position = self.position

        self.position_before(query, block)

        entered_at = self.position
        yield

        exit_at = self.position
        self.block = parent_block

        # accounting for operations added within the context
        if entered_at <= parent_position:
            parent_position += (exit_at - entered_at)

        self.position = parent_position + (exit_at - entered_at)

    @contextmanager
    def goto_after(self, query: MatchExpressionBase,
                   block: Optional[mast.Block] = None):
        """
        Enters a context to build at the point just after *query* gets matched in
        *block*.

        :arg block: Block to query the operations in. Defaults to the builder's
            block.

        Example usage::

            with builder.goto_after(Writes("%c0") & Isa(ConstantOperation)):
                # starts building after operation of form "%c0 = constant ...: ..."
                z = builder.dim(x, c0, builder.INDEX)

            # goes back to building at the builder's earlier position
        """
        parent_block = self.block
        parent_position = self.position

        self.position_after(query, block)

        entered_at = self.position
        yield

        exit_at = self.position
        self.block = parent_block

        # accounting for operations added within the context
        if entered_at <= parent_position:
            parent_position += (exit_at - entered_at)

        self.position = parent_position

    # }}}

    # {{{ standard dialect

    def addf(self, op_a: mast.SsaId, op_b: mast.SsaId, type: mast.Type,
             name: Optional[str] = None):
        op = std.AddfOperation(match=0, operand_a=op_a, operand_b=op_b, type=type)
        return self._insert_op_in_block([name], op)

    def mulf(self, op_a: mast.SsaId, op_b: mast.SsaId, type: mast.Type,
             name: Optional[str] = None):
        op = std.MulfOperation(match=0, operand_a=op_a, operand_b=op_b, type=type)
        return self._insert_op_in_block([name], op)

    def dim(self, memref_or_tensor: mast.SsaId, index: mast.SsaId,
            memref_type: Union[mast.MemRefType, mast.TensorType],
            name: Optional[str] = None):
        op = std.DimOperation(match=0, operand=memref_or_tensor, index=index,
                              type=memref_type)
        return self._insert_op_in_block([name], op)

    def index_constant(self, value: int, name: Optional[str] = None):
        op = std.ConstantOperation(match=0, value=value, type=mast.IndexType())
        return self._insert_op_in_block([name], op)

    def float_constant(self, value: float, type: mast.FloatType,
                       name: Optional[str] = None):
        op = std.ConstantOperation(match=0, value=value, type=type)
        return self._insert_op_in_block([name], op)

    # }}}

    def ret(self, values: Optional[List[mast.SsaId]] = None,
            types: Optional[List[mast.Type]] = None):

        op = std.ReturnOperation(match=0, values=values, types=types)
        self._insert_op_in_block([], op)
        self.block = None
        self.position = 0


@dataclass
class DialectBuilder:
    """
    A dialect-specific IR Builder.
    """
    core_builder: IRBuilder


class AffineBuilder(DialectBuilder):
    """
    Affine dialect ops builder.

    .. automethod:: for_
    .. automethod:: load
    .. automethod:: store
    """

    def for_(self, lower_bound: Union[int, mast.SsaId],
             upper_bound: Union[int, mast.SsaId],
             step: Optional[int] = None, indexname: Optional[str] = None):
        if indexname is None:
            indexname = self.core_builder.name_gen("i")
            index = mast.AffineSsa(indexname)

        if step is None:
            match = 0
        else:
            match = 1

        op = affine.AffineForOp(match=match,
                                index=index,
                                begin=lower_bound, end=upper_bound, step=step,
                                region=mast.Region(body=[]))

        self.core_builder._insert_op_in_block([], op)
        return op

    def load(self, memref: mast.SsaId,
             indices: Union[mast.AffineExpr, List[mast.AffineExpr]],
             memref_type: mast.MemRefType, name: Optional[str] = None):
        if isinstance(indices, mast.AffineExpr):
            indices = [indices]

        op = affine.AffineLoadOp(match=0, arg=memref,
                                 index=mast.MultiDimAffineExpr(indices),
                                 type=memref_type)
        return self.core_builder._insert_op_in_block([name], op)

    def store(self, address: mast.SsaId, memref: mast.SsaId,
              indices: Union[mast.AffineExpr, List[mast.AffineExpr]],
              memref_type: mast.MemRefType):
        if isinstance(indices, mast.AffineExpr):
            indices = [indices]

        op = affine.AffineStoreOp(match=0, addr=address, ref=memref,
                                  index=mast.MultiDimAffineExpr(indices),
                                  type=memref_type)
        self.core_builder._insert_op_in_block([], op)


# vim: fdm=marker
