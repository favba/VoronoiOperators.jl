module VoronoiOperators

using LinearAlgebra
using TensorsLite, TensorsLiteGeometry, ImmutableVectors, VoronoiMeshes
using TaskLocalValues
import SIMD as sd

export VertexToEdgeTransformation, VertexToEdgeMean, VertexToEdgeInterpolation, VertexToEdgePiecewise, VertexToEdgeArea
export EdgeToCellTransformation, EdgeToCellRingler, EdgeToCellLSq2, EdgeToCellLSq3
export CellToEdgeTransformation, CellToEdgeMean, CellToEdgeBaricentric
export VecCellToEdgeTransformation
export VecCellToEdgeMean
export CellVelocityReconstruction, CellVelocityReconstructionPerot, CellVelocityReconstructionLSq1, CellVelocityReconstructionLSq2
export CellKineticEnergyMPAS, CellKineticEnergyRingler, CellKineticEnergyVelRecon, CellKineticEnergyPerot
export GradientAtEdge, DivAtCell, CurlAtVertex, CurlAtEdge
export CellBoxFilter
export TangentialVelocityReconstruction, TangentialVelocityReconstructionThuburn, TangentialVelocityReconstructionPeixoto

include("utils.jl")

abstract type VoronoiOperator end
abstract type LinearVoronoiOperator <: VoronoiOperator end
abstract type NonLinearVoronoiOperator <: VoronoiOperator end

n_input(a::VoronoiOperator) = a.n
n_output(a::VoronoiOperator) = length(a.indices)
out_eltype(Vop::LinearVoronoiOperator, in_field, op::F = Base.identity) where {F <: Function} = Base.promote_op(*, eltype(eltype(Vop.weights)), Base.promote_op(op, eltype(in_field)))


transformation_function!(out_field::AbstractArray, in_field::AbstractArray, Vop::VoronoiOperator, op::F = Base.identity) where {F <: Function} = weighted_sum_transformation!(out_field, in_field, Vop.weights, Vop.indices, op)

transformation_function!(out_field::AbstractArray, opt_out::F, in_field::AbstractArray, Vop::VoronoiOperator, op::F2 = Base.identity) where {F <: Function, F2 <: Function} = weighted_sum_transformation!(out_field, opt_out, in_field, Vop.weights, Vop.indices, op)

function (Vop::LinearVoronoiOperator)(out_field::AbstractArray, in_field::AbstractArray, op::F = Base.identity) where {F <: Function}
    is_proper_size(in_field, n_input(Vop)) || throw(DimensionMismatch("Input array doesn't seem to be a $(name_input(Vop)) field"))
    is_proper_size(out_field, n_output(Vop)) || throw(DimensionMismatch("Output array doesn't seem to be a $(name_output(Vop)) field"))

    transformation_function!(out_field, in_field, Vop, op)

    return out_field
end

function (Vop::LinearVoronoiOperator)(out_field::AbstractArray, op_out::F, in_field::AbstractArray, op::F2 = Base.identity) where {F <: Function, F2 <: Function}
    is_proper_size(in_field, n_input(Vop)) || throw(DimensionMismatch("Input array doesn't seem to be a $(name_input(Vop)) field"))
    is_proper_size(out_field, n_output(Vop)) || throw(DimensionMismatch("Output array doesn't seem to be a $(name_output(Vop)) field"))

    transformation_function!(out_field, op_out, in_field, Vop, op)

    return out_field
end

my_similar(in_field, T, s) = similar(in_field, T, s)
my_similar(in_field, ::Type{T}, s) where {T <: Vec2Dxy} = VecArray(x = similar(in_field, nonzero_eltype(T), s), y = similar(in_field, nonzero_eltype(T), s))
my_similar(in_field, ::Type{T}, s) where {T <: Vec3D} = VecArray(x = similar(in_field, nonzero_eltype(T), s), y = similar(in_field, nonzero_eltype(T), s), z = similar(in_field, nonzero_eltype(T), s))

function (Vop::LinearVoronoiOperator)(in_field::AbstractArray, op::F = Base.identity) where {F <: Function}
    is_proper_size(in_field, n_input(Vop)) || throw(DimensionMismatch("Input array doesn't seem to be a $(name_input(Vop)) field"))
    s = construct_new_node_index(size(in_field)..., n_output(Vop))
    out_field = my_similar(in_field, out_eltype(Vop, in_field, op), s)
    return Vop(out_field, in_field, op)
end

include("node_transformations.jl")
include("edge_to_cell_transformations.jl")
include("cell_vector_to_edge_transformations.jl")
include("cell_velocity_reconstruction.jl")
include("kinetic_energy_reconstruction.jl")
include("differential_operators.jl")
include("filtering_operators.jl")
include("tangential_velocity_reconstruction.jl")

end
