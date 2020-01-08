Creating a Custom Dialect
=========================

One of MLIR's most powerful features is being able to define custom dialects. Parsing
custom dialects in pyMLIR is done by adding them to the ``dialects`` field of the
mlir parser, as in the following snippet:

.. code-block:: python

  import mlir

  # Load and verify dialect
  dialect = mlir.Dialect('customdialect.lark', 'NAME')

  # Add dialect to the parser
  m = mlir.parse_path('/path/to/file.mlir', dialects=[dialect])

Dialects have names, which are defined upon parsing. In this document, we use ``NAME`` as
the dialect's name.

To be used, a dialect must implement a ``.lark`` file that contains at least the following
two fields::

    NAME_dialect_operations : op1 | op2 | op3  // ...
    NAME_dialect_types :
    // No types are defined, so "NAME_dialect_types" is empty

pyMLIR will then detect the two fields and inject them into the parser.

Advanced Dialect Behavior
-------------------------

In order to extend custom behavior in the dialect, you can extend the ``mlir.Dialect`` class.