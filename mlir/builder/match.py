"""Operation query interface."""

import mlir.astnodes as ast
from mlir.visitors import NodeVisitor
from typing import List, Union
from dataclasses import dataclass


__doc__ = """
.. currentmodule:: mlir.builder.match

Querying expressions
====================

.. autoclass:: All
.. autoclass:: And
.. autoclass:: Or
.. autoclass:: Not
.. autoclass:: Reads
.. autoclass:: Writes
.. autoclass:: Isa
"""


class SsaIdCollector(NodeVisitor):
    """
    A visitor to collect all the visited :class:`SsaId`'s in a node.
    """
    def __init__(self):
        self.visited_ssas = []

    def visit_SsaId(self, ssa: ast.SsaId):
        self.visited_ssas.append(ssa)


class MatchExpressionBase:
    def __call__(self, op: ast.Operation) -> bool:
        raise NotImplementedError()

    def __and__(self, other: "MatchExpressionBase") -> "And":
        return And([self, other])

    def __or__(self, other: "MatchExpressionBase") -> "Or":
        return Or([self, other])


class All(MatchExpressionBase):
    """
    Matches with all nodes.
    """
    def __call__(self, op: ast.Operation) -> bool:
        return True


@dataclass
class And(MatchExpressionBase):
    """
    Matches if all its children match.
    """
    children: List[MatchExpressionBase]

    def __call__(self, op: ast.Operation) -> bool:
        return all(ch(op) for ch in self.children)


@dataclass
class Or(MatchExpressionBase):
    """
    Matches if any of its children match.
    """
    children: List[MatchExpressionBase]

    def __call__(self, op: ast.Operation) -> bool:
        return any(ch(op) for ch in self.children)


@dataclass
class Not(MatchExpressionBase):
    """
    Matches if the child does not match.
    """
    child: MatchExpressionBase

    def __call__(self, op: ast.Operation) -> bool:
        return not self.child(op)


@dataclass
class Reads(MatchExpressionBase):
    """
    Matches the variables read by the operation.
    """
    name: Union[str, ast.SsaId]

    def __call__(self, op: ast.Operation) -> bool:
        collector = SsaIdCollector()
        collector.visit(op.op)
        visited_ssas = collector.visited_ssas
        return any((ssa == self.name) or (ssa.dump() == self.name)
                   for ssa in visited_ssas)


@dataclass
class Writes(MatchExpressionBase):
    """
    Matches the variable names written by the operation.
    """
    name: Union[str, ast.SsaId]

    def __call__(self, op: ast.Operation) -> bool:
        return any((ssa == self.name) or (ssa.dump() == self.name)
                   for ssa in op.result_list)


@dataclass
class Isa(MatchExpressionBase):
    """
    Matches the operation's type.
    """
    type: type

    def __call__(self, op: ast.Operation) -> bool:
        return isinstance(op.op, self.type)

# vim: fdm=marker
