""" Implementation of the Linalg dialect. """

import inspect
import sys
import mlir.astnodes as mast
from mlir.dialect import Dialect, DialectOp, is_op
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class LinalgBatchMatmul(DialectOp):
    a_id: mast.SsaId
    b_id: mast.SsaId
    a_type: mast.Type
    b_type: mast.Type
    c_id: mast.SsaId
    c_type: mast.Type
    out_type: Optional[mast.Type] = None

    _syntax_ = [("linalg.batch_matmul"
                 " ins( {a_id.ssa_id} , {b_id.ssa_id} : {a_type.type} , {b_type.type} )"
                 " outs( {c_id.ssa_id} : {c_type.type} )"),
                ("linalg.batch_matmul"
                 " ins( {a_id.ssa_id} , {b_id.ssa_id} : {a_type.type} , {b_type.type} )"
                 " init( {c_id.ssa_id} : {c_type.type} ) -> {out_type.type}")]


@dataclass
class LinalgConvW(DialectOp):
    in_id: mast.SsaId
    filter_id: mast.SsaId
    in_type: mast.Type
    filter_type: mast.Type
    out_id: mast.SsaId
    out_type: mast.Type

    _syntax_ = [("linalg.conv_1d"
                 " ins( {in_id.ssa_id} , {filter_id.ssa_id} : {in_type.type} , {filter_type.type} )"
                 " outs( {out_id.ssa_id} : {out_type.type} )")]


@dataclass
class LinalgConvHW(DialectOp):
    in_id: mast.SsaId
    filter_id: mast.SsaId
    in_type: mast.Type
    filter_type: mast.Type
    out_id: mast.SsaId
    out_type: mast.Type

    _syntax_ = [("linalg.conv_2d"
                 " ins( {in_id.ssa_id} , {filter_id.ssa_id} : {in_type.type} , {filter_type.type} )"
                 " outs( {out_id.ssa_id} : {out_type.type} )")]


@dataclass
class LinalgConvDHW(DialectOp):
    in_id: mast.SsaId
    filter_id: mast.SsaId
    in_type: mast.Type
    filter_type: mast.Type
    out_id: mast.SsaId
    out_type: mast.Type

    _syntax_ = [("linalg.conv_3d"
                 " ins( {in_id.ssa_id} , {filter_id.ssa_id} : {in_type.type} , {filter_type.type} )"
                 " outs( {out_id.ssa_id} : {out_type.type} )")]


@dataclass
class LinalgConv(DialectOp):
    in_id: mast.SsaId
    filter_id: mast.SsaId
    in_type: mast.Type
    filter_type: mast.Type
    out_id: mast.SsaId
    out_type: mast.Type
    attr: Optional[mast.Attribute] = None

    _syntax_ = [("linalg.conv( {in_id.ssa_id} , {filter_id.ssa_id} , {out_id.ssa_id} ) "
                "{attr.attribute_value} : {in_type.type} , {filter_type.type} , {out_type.type}"),
                ("linalg.conv( {in_id.ssa_id} , {filter_id.ssa_id} , {out_id.ssa_id} ) "
                " : {in_type.type} , {filter_type.type} , {out_type.type}")]


@dataclass
class LinalgCopy(DialectOp):
    a_id: mast.SsaId
    b_id: mast.SsaId
    a_type: mast.Type
    b_type: mast.Type
    attr: Optional[mast.Attribute] = None

    _syntax_ = [("linalg.copy( {a_id.ssa_id} , {b_id.ssa_id} ) "
                "{attr.attribute_value} : {a_type.type} , {b_type.type}"),
                ("linalg.copy( {a_id.ssa_id} , {b_id.ssa_id} ) "
                " : {a_type.type} , {b_type.type}")]


@dataclass
class LinalgDot(DialectOp):
    in_a_id: mast.SsaId
    in_b_id: mast.SsaId
    in_a_type: mast.Type
    in_b_type: mast.Type
    out_id: mast.SsaId
    out_type: mast.Type

    _syntax_ = [("linalg.dot"
                 " ins( {in_a_id.ssa_id} , {in_b_id.ssa_id} : {in_a_type.type} , {in_b_type.type} )"
                 " outs( {out_id.ssa_id} : {out_type.type} )")]


@dataclass
class LinalgFill(DialectOp):
    output_id: mast.SsaId
    value_id: mast.SsaId
    output_type: mast.Type
    value_type: mast.Type
    attr: Optional[mast.Attribute] = None

    _syntax_ = [("linalg.fill( {output_id.ssa_id} , {value_id.ssa_id} ) "
                "{attr.attribute_value} : {output_type.type} , {value_type.type}"),
                ("linalg.fill( {output_id.ssa_id} , {value_id.ssa_id} ) "
                " : {output_type.type} , {value_type.type}")]


@dataclass
class LinalgGeneric(DialectOp):
    inargs: List[mast.SsaId]
    in_types: List[mast.Type]
    region: mast.Region
    outargs: Optional[List[mast.SsaId]] = None
    out_types: Optional[List[mast.Type]] = None
    init_args: Optional[List[mast.SsaId]] = None
    init_types: Optional[List[mast.Type]] = None
    out_type: Optional[mast.Type] = None
    attr: Optional[mast.Attribute] = None

    _syntax_ = [("linalg.generic {attr.attribute_value} "
                 " ins( {inargs.ssa_id_list} : {in_types.type_list_no_parens} )"
                 " outs( {outargs.ssa_id_list} : {out_types.type_list_no_parens} )"
                 " {region.region}"),
                ("linalg.generic {attr.attribute_value} "
                 " ins( {inargs.ssa_id_list} : {in_types.type_list_no_parens} )"
                 " init( {init_args.ssa_id_list} : {init_types.type_list_no_parens} )"
                 " {region.region} -> {out_type.type}")]


@dataclass
class LinalgIndexedGeneric(DialectOp):
    inargs: List[mast.SsaId]
    in_types: List[mast.Type]
    region: mast.Region
    outargs: Optional[List[mast.SsaId]] = None
    out_types: Optional[List[mast.Type]] = None
    init_args: Optional[List[mast.SsaId]] = None
    init_types: Optional[List[mast.Type]] = None
    out_type: Optional[mast.Type] = None
    attr: Optional[mast.Attribute] = None

    _syntax_ = [("linalg.indexed_generic {attr.attribute_value} "
                 " ins( {inargs.ssa_id_list} : {in_types.type_list_no_parens} )"
                 " outs( {outargs.ssa_id_list} : {out_types.type_list_no_parens} )"
                 " {region.region}"),
                ("linalg.indexed_generic {attr.attribute_value} "
                 " ins( {inargs.ssa_id_list} : {in_types.type_list_no_parens} )"
                 " init( {init_args.ssa_id_list} : {init_types.type_list_no_parens} )"
                 " {region.region} -> {out_type.type}")]


@dataclass
class LinalgRange(DialectOp):
    min_id: mast.SsaId
    max_id: mast.SsaId
    step_id: mast.SsaId
    out_type: mast.Type
    attr: Optional[mast.Attribute] = None

    _syntax_ = [("linalg.range {min_id.ssa_id} : {max_id.ssa_id} : {step_id.ssa_id}"
                 " {attr.attribute_value} : {out_type.type}"),
                ("linalg.range {min_id.ssa_id} : {max_id.ssa_id} : {step_id.ssa_id}"
                 " : {out_type.type}")]


@dataclass
class LinalgReshape(DialectOp):
    src_id: mast.SsaId
    src_type: mast.MemRefType
    result_type: mast.MemRefType
    reassociation: Optional[List[mast.AffineMap]] = None
    attr: Optional[mast.Attribute] = None

    _syntax_ = [("linalg.reshape {src_id.ssa_id}"
                 " [ {reassociation.affine_map_list} ] "
                 " {attr.attribute_value} "
                 " : {src_type.memref_type} into {result_type.memref_type}"),
                ("linalg.reshape {src_id.ssa_id}"
                 " [ ] "
                 " {attr.attribute_value} "
                 " : {src_type.memref_type} into {result_type.memref_type}"),
                ("linalg.reshape {src_id.ssa_id}"
                 " [ {reassociation.affine_map_list} ] "
                 " : {src_type.memref_type} into {result_type.memref_type}"),
                ("linalg.reshape {src_id.ssa_id}"
                 " [ ] "
                 " : {src_type.memref_type} into {result_type.memref_type}")]


@dataclass
class LinalgSlice(DialectOp):
    view_id: mast.SsaId
    indexing_ids: List[mast.SsaId]
    view_type: mast.Type
    indexing_types: List[mast.Type]
    result_type: mast.Type

    _syntax_ = ("linalg.slice {view_id.ssa_id} [ {indexing_ids.ssa_id_list} ]"
                " : {view_type.type} , {indexing_types.type_list_no_parens} "
                " , {result_type.type}")


@dataclass
class TensorReshape(DialectOp):
    src_id: mast.SsaId
    src_type: mast.MemRefType
    result_type: mast.MemRefType
    reassociation: Optional[List[mast.AffineMap]] = None
    attr: Optional[mast.Attribute] = None

    _syntax_ = [("linalg.tensor_reshape {src_id.ssa_id}"
                 " [ {reassociation.affine_map_list} ] "
                 " {attr.attribute_value} "
                 " : {src_type.tensor_type} into {result_type.tensor_type}"),
                ("linalg.tensor_reshape {src_id.ssa_id}"
                 " [ ] "
                 " {attr.attribute_value} "
                 " : {src_type.tensor_type} into {result_type.tensor_type}"),
                ("linalg.tensor_reshape {src_id.ssa_id}"
                 " [ {reassociation.affine_map_list} ] "
                 " : {src_type.tensor_type} into {result_type.tensor_type}"),
                ("linalg.tensor_reshape {src_id.ssa_id}"
                 " [ ] "
                 " : {src_type.tensor_type} into {result_type.tensor_type}")]


@dataclass
class LinalgYield(DialectOp):
    operand_ids: List[mast.SsaId]
    operand_types: List[mast.Type]

    _syntax_ = ("linalg.yield {operand_ids.ssa_id_list}"
                " : {operand_types.type_list_no_parens}")


@dataclass
class LinalgMatmul(DialectOp):
    a_id: mast.SsaId
    b_id: mast.SsaId
    a_type: mast.Type
    b_type: mast.Type
    c_id: mast.SsaId
    c_type: mast.Type
    out_type: Optional[mast.Type] = None

    _syntax_ = [("linalg.matmul"
                 " ins( {a_id.ssa_id} , {b_id.ssa_id} : {a_type.type} , {b_type.type} )"
                 " outs( {c_id.ssa_id} : {c_type.type} )"),
                ("linalg.matmul"
                 " ins( {a_id.ssa_id} , {b_id.ssa_id} : {a_type.type} , {b_type.type} )"
                 " init( {c_id.ssa_id} : {c_type.type} )  -> {out_type.type}")]


@dataclass
class LinalgMatvec(DialectOp):
    a_id: mast.SsaId
    b_id: mast.SsaId
    a_type: mast.Type
    b_type: mast.Type
    c_id: mast.SsaId
    c_type: mast.Type

    _syntax_ = [("linalg.matvec"
                 " ins( {a_id.ssa_id} , {b_id.ssa_id} : {a_type.type} , {b_type.type} )"
                 " outs( {c_id.ssa_id} : {c_type.type} )")]


# Inspect current module to get all classes defined above
linalg = Dialect("linalg", ops=[m[1] for m in inspect.getmembers(
    sys.modules[__name__], lambda obj: is_op(obj, __name__))])
