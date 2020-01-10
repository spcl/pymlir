""" Contains classes that parse MLIR files """

import itertools
from lark import Lark, Tree
import os
import sys
from typing import List, Optional, TextIO

from mlir.parser_transformer import TreeToMlir
from mlir.dialect import Dialect
from mlir import astnodes as mast

# Load the MLIR EBNF syntax to memory once
_MLIR_LARK = None
_DIALECT_LARKS: List[Dialect] = []


def _lazy_load():
    """
    Loads the Lark EBNF files (MLIR and default dialects) into memory upon
    first use.
    """
    global _MLIR_LARK
    global _DIALECT_LARKS

    # Lazily load the MLIR EBNF file and the dialects
    if _MLIR_LARK is None:
        # Find path to files
        mlir_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), 'lark')
        dialects_path = os.path.join(mlir_path, 'dialects')

        with open(os.path.join(mlir_path, 'mlir.lark'), 'r') as fp:
            _MLIR_LARK = fp.read()

        _DIALECT_LARKS = []
        for dialect in os.listdir(dialects_path):
            if dialect.endswith('.lark'):
                # Cut the last extension characters
                dialect_name = os.path.basename(dialect)[:-5]
                _DIALECT_LARKS.append(
                    Dialect(
                        os.path.join(dialects_path, dialect), dialect_name))


def parse_string(code: str,
                 dialects: Optional[List[Dialect]] = None) -> mast.Module:
    """
    Parses a string representing code in MLIR, returning the top-level AST node.
    :param code: A code string in MLIR format.
    :param dialects: An optional list of additional dialects to load (in
                     addition to the built-in dialects).
    :return: A module node representing the root of the AST.
    """
    # Lazy-load (if necessary) the Lark files
    _lazy_load()

    # Check validity of given dialects
    dialects = dialects or []
    builtin_names = [dialect.name for dialect in _DIALECT_LARKS]
    additional_names = [dialect.name for dialect in dialects]
    dialect_set = set(builtin_names) | set(additional_names)
    if len(dialect_set) != (len(dialects) + len(_DIALECT_LARKS)):
        raise NameError(
            'Additional dialect already exists (built-in dialects: %s, given '
            'dialects: %s)' % (builtin_names, additional_names))

    # Create a parser from the MLIR EBNF file, default dialects, and additional
    # dialects if exist
    parser_src = _MLIR_LARK + ''
    op_expr = 'pymlir_dialect_ops: '
    type_expr = 'pymlir_dialect_types: '
    first = True
    for dialect in itertools.chain(_DIALECT_LARKS, dialects):
        parser_src += dialect.contents
        if not first:
            op_expr += '| '
            type_expr += '| '
        first = False
        op_expr += dialect.name + '_dialect_operations '
        type_expr += dialect.name + '_dialect_types '

    parser_src += op_expr + '\n' + type_expr

    # Create parser
    parser = Lark(parser_src, parser='earley')

    # Parse code and return result
    transformer = TreeToMlir()
    tree = parser.parse(code)
    root_node = transformer.transform(tree)

    # If the root node is a function/definition or a list thereof, return
    # a top-level module
    if not isinstance(root_node, mast.Module):
        if isinstance(root_node, Tree) and root_node.data == 'start':
            return mast.Module([root_node])
        return mast.Module(root_node)
    return root_node


def parse_file(file: TextIO,
               dialects: Optional[List[Dialect]] = None) -> mast.Node:
    """
    Parses an MLIR file from a given Python file-like object, returning the
    top-level AST node.
    :param file: Python file-like I/O object in text mode.
    :param dialects: An optional list of additional dialects to load (in
                     addition to the built-in dialects).
    :return: A module node representing the root of the AST.
    """
    return parse_string(file.read(), dialects)


def parse_path(file_path: str,
               dialects: Optional[List[Dialect]] = None) -> mast.Node:
    """
    Parses an MLIR file from a given filename, returning the top-level AST node.
    :param file_path: Path to file to parse.
    :param dialects: An optional list of additional dialects to load (in
                     addition to the built-in dialects).
    :return: A module node representing the root of the AST.
    """
    with open(file_path, 'r') as fp:
        return parse_file(fp, dialects)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('USAGE: python -m mlir.parser <MLIR FILE> [DIALECT PATHS...]')
        exit(1)

    additional_dialects = []
    for dialect_name in sys.argv[2:]:
        additional_dialects.append(
            Dialect(dialect_name,
                    os.path.basename(dialect_name)[:-5]))

    print(parse_path(sys.argv[1], dialects=additional_dialects).pretty())
