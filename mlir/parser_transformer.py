from lark import v_args, Transformer
from lark.exceptions import GrammarError
from lark.visitors import Discard
from mlir import astnodes


class TreeToMlir(Transformer):
    def _call_userfunc(self, tree, new_children=None):
        # Assumes tree is already transformed
        children = new_children if new_children is not None else tree.children
        try:
            f = getattr(self, tree.data)
        except AttributeError:
            return self.__default__(tree.data, children, tree.meta)
        else:
            try:
                wrapper = getattr(f, 'visit_wrapper', None)
                if wrapper is not None:
                    return f.visit_wrapper(f, tree.data, children, tree.meta)
                else:
                    return f(children)
            except (GrammarError, Discard):
                raise
    ###############################################################
    # Low-level literal syntax
    digit = lambda self, val: int(val[0])
    digits = lambda self, val: int(val[0])
    hex_digit = lambda self, val: str(val[0])
    hex_digits = lambda self, val: str(val[0])
    letter = lambda self, val: str(val[0])
    letters = lambda self, val: str(val[0])
    id_punct = lambda self, val: str(val[0])
    underscore = lambda self, val: str(val[0])
    true = lambda self, _: True
    false = lambda self, _: False
    id_chars = lambda self, val: str(val[0])
    inttype_width = lambda self, val: int(val[0])
    dimension = astnodes.Dimension.from_lark

    # Literals
    @v_args(inline=True)
    def decimal_literal(self, *digits):
        return int(''.join(str(d) for d in digits))

    @v_args(inline=True)
    def hexadecimal_literal(self, *digits):
        return '0x' + ''.join(digits)

    negated_integer_literal = lambda self, value: -value[0]
    float_literal = lambda self, value: float(value[0])

    @v_args(inline=True)
    def string_literal(self, s):
        return astnodes.StringLiteral(s[1:-1].replace('\\"', '"'))

    @v_args(inline=True)
    def bare_id(self, *elements):
        return ''.join(str(s) for s in elements)

    @v_args(inline=True)
    def suffix_id(self, *suffix):
        return ''.join(str(s) for s in suffix)

    ###############################################################
    # MLIR Identifiers

    ssa_id = astnodes.SsaId.from_lark
    symbol_ref_id = astnodes.SymbolRefId.from_lark
    block_id = astnodes.BlockId.from_lark
    type_alias = astnodes.TypeAlias.from_lark
    attribute_alias = astnodes.AttrAlias.from_lark
    map_or_set_id = astnodes.MapOrSetId.from_lark

    ###############################################################
    # MLIR Types

    none_type = astnodes.NoneType.from_lark
    F16 = lambda self, tok: astnodes.FloatTypeEnum("f16")
    BF16 = lambda self, tok: astnodes.FloatTypeEnum("bf16")
    F32 = lambda self, tok: astnodes.FloatTypeEnum("f32")
    F64 = lambda self, tok: astnodes.FloatTypeEnum("f64")
    float_type = astnodes.FloatType.from_lark
    index_type = astnodes.IndexType.from_lark
    integer_type = astnodes.IntegerType.from_lark
    complex_type = astnodes.ComplexType.from_lark
    tuple_type = astnodes.TupleType.from_lark
    vector_type = astnodes.VectorType.from_lark
    ranked_tensor_type = astnodes.RankedTensorType.from_lark
    unranked_tensor_type = lambda self, value: astnodes.UnrankedTensorType(value[1])  # gets rid of literal "*x"
    ranked_memref_type = astnodes.RankedMemRefType.from_lark
    unranked_memref_type = astnodes.UnrankedMemRefType.from_lark
    opaque_dialect_item = astnodes.OpaqueDialectType.from_lark
    pretty_dialect_item = astnodes.PrettyDialectType.from_lark
    function_type = astnodes.FunctionType.from_lark
    strided_layout = astnodes.StridedLayout.from_lark

    ###############################################################
    # MLIR Attributes

    array_attribute = astnodes.ArrayAttr
    bool_attribute = astnodes.BoolAttr.from_lark
    dictionary_attribute = astnodes.DictionaryAttr
    dense_elements_attribute = astnodes.DenseElementsAttr.from_lark
    opaque_elements_attribute = astnodes.OpaqueElementsAttr.from_lark
    sparse_elements_attribute = astnodes.SparseElementsAttr.from_lark
    float_attribute = astnodes.FloatAttr.from_lark
    integer_attribute = astnodes.IntegerAttr.from_lark
    integer_set_attribute = astnodes.IntSetAttr.from_lark
    string_attribute = astnodes.StringAttr.from_lark
    symbol_ref_attribute = astnodes.SymbolRefAttr
    type_attribute = astnodes.TypeAttr.from_lark
    unit_attribute = astnodes.UnitAttr.from_lark

    dependent_attribute_entry = astnodes.AttributeEntry.from_lark
    dialect_attribute_entry = astnodes.DialectAttributeEntry.from_lark
    attribute_dict = astnodes.AttributeDict

    ###############################################################
    # Operations

    op_result = astnodes.OpResult.from_lark
    location = astnodes.FileLineColLoc.from_lark
    operation = astnodes.Operation.from_lark
    generic_operation = astnodes.GenericOperation.from_lark
    custom_operation = astnodes.CustomOperation.from_lark

    ###############################################################
    # Blocks, regions, modules, functions

    def block_label(self, value):
        if value[1] is None:
            arg_ids, argtypes = None, None
        else:
            arg_ids, argtypes = list(zip(*value[1]))
            return astnodes.BlockLabel(value[0], arg_ids, argtypes)
    block = astnodes.Block.from_lark
    region = astnodes.Region
    module = astnodes.Module.from_lark
    function = astnodes.Function.from_lark
    named_argument = astnodes.NamedArgument.from_lark

    ###############################################################
    # (semi-)Affine expressions, maps, and integer sets

    dim_and_symbol_id_lists = astnodes.DimAndSymbolList.from_lark
    dim_and_symbol_use_list = astnodes.DimAndSymbolList.from_lark

    affine_expr = astnodes.AffineExpr.from_lark
    semi_affine_expr = astnodes.SemiAffineExpr.from_lark
    multi_dim_affine_expr = astnodes.MultiDimAffineExpr.from_lark
    multi_dim_semi_affine_expr = astnodes.MultiDimSemiAffineExpr.from_lark

    affine_constraint_ge = astnodes.AffineConstraintGreaterEqual.from_lark
    affine_constraint_eq = astnodes.AffineConstraintEqual.from_lark

    affine_map_inline = astnodes.AffineMap.from_lark
    semi_affine_map_inline = astnodes.SemiAffineMap.from_lark
    integer_set_inline = astnodes.IntSet.from_lark

    affine_neg = astnodes.AffineNeg.from_lark
    semi_affine_neg = astnodes.AffineNeg.from_lark
    affine_parens = astnodes.AffineParens.from_lark
    semi_affine_parens = astnodes.AffineParens.from_lark
    affine_symbol_explicit = astnodes.AffineExplicitSymbol.from_lark
    semi_affine_symbol_explicit = astnodes.AffineExplicitSymbol.from_lark
    affine_add = astnodes.AffineAdd.from_lark
    semi_affine_add = astnodes.AffineAdd.from_lark
    affine_sub = astnodes.AffineSub.from_lark
    semi_affine_sub = astnodes.AffineSub.from_lark
    affine_mul = astnodes.AffineMul.from_lark
    semi_affine_mul = astnodes.AffineMul.from_lark
    affine_floordiv = astnodes.AffineFloorDiv.from_lark
    semi_affine_floordiv = astnodes.AffineFloorDiv.from_lark
    affine_ceildiv = astnodes.AffineCeilDiv.from_lark
    semi_affine_ceildiv = astnodes.AffineCeilDiv.from_lark
    affine_mod = astnodes.AffineMod.from_lark
    semi_affine_mod = astnodes.AffineMod.from_lark

    ###############################################################
    # Top-level definitions

    type_alias_def = astnodes.TypeAliasDef.from_lark
    affine_map_def = astnodes.AffineMapDef.from_lark
    semi_affine_map_def = astnodes.SemiAffineMapDef.from_lark
    integer_set_def = astnodes.IntSetDef.from_lark
    attribute_alias_def = astnodes.AttrAliasDef.from_lark

    ###############################################################
    # List types
    bare_id_list = list
    ssa_id_list = list
    ssa_use_list = list
    op_result_list = list
    successor_list = list
    ssa_id_and_type_list = tuple
    ssa_use_and_type_list = list
    stride_list = list
    dimension_list_ranked = list
    static_dimension_list = list
    pretty_dialect_item_body = list
    type_list_no_parens = list
    affine_constraint_conjunction = list
    function_result_list_no_parens = list
    multi_dim_affine_expr_no_parens = list
    multi_dim_semi_affine_expr_no_parens = list
    dim_id_list = list
    symbol_id_list = list
    dim_use_list = list
    symbol_use_list = list
    operation_list = list
    argument_list = list
    definition_list = list
    function_list = list
    module_list = list
    definition_and_function_list = tuple
    definition_and_module_list = tuple

    ###############################################################
    # Composite types that should be reduced to sub-types
    bool_literal = lambda self, value: value[0]
    integer_literal = lambda self, value: value[0]
    constant_literal = lambda self, value: value[0]
    dimension_list = lambda self, value: value[0]
    ssa_use = lambda self, value: value[0]
    vector_element_type = lambda self, value: value[0]
    tensor_memref_element_type = lambda self, value: value[0]
    tensor_type = lambda self, value: value[0]
    memref_type = lambda self, value: value[0]
    standard_type = lambda self, value: value[0]
    dialect_type = lambda self, value: value[0]
    non_function_type = lambda self, value: value[0]
    type = lambda self, value: value[0]
    type_list_parens = lambda self, value: (value[0] if value else [])
    function_result = lambda self, value: value[0]
    function_result_type = lambda self, value: value[0]
    standard_attribute = lambda self, value: value[0]
    attribute_value = lambda self, value: value[0]
    dialect_attribute = lambda self, value: value[0]
    attribute_entry = lambda self, value: value[0]
    trailing_type = lambda self, value: value[0]
    trailing_location = lambda self, value: value[0]
    function_result_list_parens = lambda self, value: (value[0] if value else [])
    symbol_or_const = lambda self, value: value[0]
    affine_map = lambda self, value: value[0]
    semi_affine_map = lambda self, value: value[0]
    integer_set = lambda self, value: value[0]
    affine_literal = lambda self, value: value[0]
    semi_affine_literal = lambda self, value: value[0]
    affine_ssa = lambda self, value: value[0]
    affine_symbol = lambda self, value: value[0]
    semi_affine_symbol = lambda self, value: value[0]

    ###############################################################
    # MLIR file

    def only_functions_and_definitions_file(self, defns_and_fns):
        assert isinstance(defns_and_fns, list)
        assert all(isinstance(el, tuple) for el in defns_and_fns)
        defns = sum([defns for defns, fns in defns_and_fns], [])
        fns = sum([fns for defns, fns in defns_and_fns], [])
        return astnodes.MLIRFile(defns, astnodes.Module(None, None, astnodes.Region(fns)))


    def mlir_file_as_definition_and_module_list(self, defns_and_mods):
        assert isinstance(defns_and_mods, list)
        assert all(isinstance(el, tuple) for el in defns_and_mods)
        defns = sum([defns for defns, mods in defns_and_mods], [])
        mods = sum([mods for defns, mods in defns_and_mods], [])
        if len(mods) == 1:
            module = mods[0]
        else:
            module = astnodes.Module(None, None, astnodes.Region(mods))
        return astnodes.MLIRFile(defns, module)

    # Dialect ops and types are appended to this list via "setattr"

    def optional(self, value):
        assert isinstance(value, list)
        assert len(value) in [0, 1]
        return value[0] if value else None
