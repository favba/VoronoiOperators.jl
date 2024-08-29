module VoronoiOperators

using TensorsLite, TensorsLiteGeometry, ImmutableVectors, VoronoiMeshDataStruct
using TaskLocalValues

export VertexToEdgeTransformation, VertexToEdgeMean, VertexToEdgeInterpolation, VertexToEdgePiecewise, VertexToEdgeArea
export CellToEdgeTransformation, CellToEdgeMean, CellToEdgeBaricentric
export VecCellToEdgeTransformation
export VecCellToEdgeMean
export CellVelocityReconstruction,CellVelocityReconstructionPerot
export CellKineticEnergyMPAS, CellKineticEnergyRingler, CellKineticEnergyVelRecon, CellKineticEnergyPerot
export GradientAtEdge, DivAtCell
export CellBoxFilter

include("utils.jl")

abstract type VoronoiOperator end

n_input(a::VoronoiOperator) = a.n
n_output(a::VoronoiOperator) = length(a.indices)

abstract type NonLinearVoronoiOperator <: VoronoiOperator end

abstract type LinearVoronoiOperator <: VoronoiOperator end

function (Vop::LinearVoronoiOperator)(out_field::AbstractArray,in_field::AbstractArray)
    is_proper_size(in_field, n_input(Vop)) || throw(DimensionMismatch("Input array doesn't seem to be a $(name_input) field"))
    is_proper_size(out_field, n_output(Vop)) || throw(DimensionMismatch("Output array doesn't seem to be a $(name_output(Vop)) field"))

    weighted_sum_transformation!(out_field, in_field, Vop.weights, Vop.indices)
    
    return out_field
end

function (Vop::LinearVoronoiOperator)(out_field::AbstractArray, op::F, in_field::AbstractArray) where F<:Function
    is_proper_size(in_field, n_input(Vop)) || throw(DimensionMismatch("Input array doesn't seem to be a $(name_input(Vop)) field"))
    is_proper_size(out_field, n_output(Vop)) || throw(DimensionMismatch("Output array doesn't seem to be a $(name_output(Vop)) field"))

    weighted_sum_transformation!(out_field, op, in_field, Vop.weights, Vop.indices)
    
    return out_field
end

function (Vop::LinearVoronoiOperator)(out_field::AbstractArray, in_field::AbstractArray, op::F) where F<:Function
    is_proper_size(in_field, n_input(Vop)) || throw(DimensionMismatch("Input array doesn't seem to be a $(name_input(Vop)) field"))
    is_proper_size(out_field, n_output(Vop)) || throw(DimensionMismatch("Output array doesn't seem to be a $(name_output(Vop)) field"))

    weighted_sum_transformation!(out_field, in_field, op, Vop.weights, Vop.indices)
    
    return out_field
end

function (Vop::LinearVoronoiOperator)(out_field::AbstractArray, op::F, in_field::AbstractArray, op2::F2) where {F<:Function,F2<:Function}
    is_proper_size(in_field, n_input(Vop)) || throw(DimensionMismatch("Input array doesn't seem to be a $(name_input(Vop)) field"))
    is_proper_size(out_field, n_output(Vop)) || throw(DimensionMismatch("Output array doesn't seem to be a $(name_output(Vop)) field"))

    weighted_sum_transformation!(out_field, op, in_field, op2, Vop.weights, Vop.indices)
    
    return out_field
end

function (Vop::VoronoiOperator)(in_field::AbstractArray)
    is_proper_size(in_field, n_input(Vop)) || throw(DimensionMismatch("Input array doesn't seem to be a $(name_input(Vop)) field"))
    s = construct_new_node_index(size(in_field)..., n_output(Vop))
    out_field = similar(in_field, Base.promote_op(*, eltype(eltype(Vop.weights)), eltype(in_field)), s)
    return Vop(out_field, in_field)
end

function (Vop::LinearVoronoiOperator)(in_field::AbstractArray, op::F) where F<:Function
    is_proper_size(in_field, n_input(Vop)) || throw(DimensionMismatch("Input array doesn't seem to be a $(name_input(Vop)) field"))
    s = construct_new_node_index(size(in_field)..., n_output(Vop))
    out_field = similar(in_field, Base.promote_op(*, eltype(eltype(Vop.weights)), Base.promote_op(op,eltype(in_field))), s)
    return Vop(out_field, in_field, op)
end

include("node_transformations.jl")
include("cell_vector_to_edge_transformations.jl")
include("cell_velocity_reconstruction.jl")
include("kinetic_energy_reconstruction.jl")
include("differential_operators.jl")
include("filtering_operators.jl")

end
