module VoronoiOperators

using TensorsLite, TensorsLiteGeometry, ImmutableVectors, VoronoiMeshDataStruct
using TaskLocalValues
import SIMD as sd

export VertexToEdgeTransformation, VertexToEdgeMean, VertexToEdgeInterpolation, VertexToEdgePiecewise, VertexToEdgeArea
export CellToEdgeTransformation, CellToEdgeMean, CellToEdgeBaricentric
export VecCellToEdgeTransformation
export VecCellToEdgeMean
export CellVelocityReconstruction, CellVelocityReconstructionPerot
export CellKineticEnergyMPAS, CellKineticEnergyRingler, CellKineticEnergyVelRecon, CellKineticEnergyPerot
export GradientAtEdge, DivAtCell
export CellBoxFilter
export TangentialVelocityReconstructionThuburn

include("utils.jl")

abstract type VoronoiOperator end

n_input(a::VoronoiOperator) = a.n
n_output(a::VoronoiOperator) = length(a.indices)

abstract type NonLinearVoronoiOperator <: VoronoiOperator end

abstract type LinearVoronoiOperator <: VoronoiOperator end

function (Vop::LinearVoronoiOperator)(out_field::AbstractArray, in_field::AbstractArray, op::F = Base.identity) where {F <: Function}
    is_proper_size(in_field, n_input(Vop)) || throw(DimensionMismatch("Input array doesn't seem to be a $(name_input) field"))
    is_proper_size(out_field, n_output(Vop)) || throw(DimensionMismatch("Output array doesn't seem to be a $(name_output(Vop)) field"))

    weighted_sum_transformation!(out_field, in_field, Vop.weights, Vop.indices, op)

    return out_field
end

function (Vop::LinearVoronoiOperator)(out_field::AbstractArray, op_out::F, in_field::AbstractArray, op::F2 = Base.identity) where {F <: Function, F2 <: Function}
    is_proper_size(in_field, n_input(Vop)) || throw(DimensionMismatch("Input array doesn't seem to be a $(name_input(Vop)) field"))
    is_proper_size(out_field, n_output(Vop)) || throw(DimensionMismatch("Output array doesn't seem to be a $(name_output(Vop)) field"))

    weighted_sum_transformation!(out_field, op_out, in_field, Vop.weights, Vop.indices, op)

    return out_field
end

my_similar(in_field, T, s) = similar(in_field, T, s)
my_similar(in_field, ::Type{T}, s) where {T <: Vec2Dxy} = VecArray(x = similar(in_field, nonzero_eltype(T), s), y = similar(in_field, nonzero_eltype(T), s))
my_similar(in_field, ::Type{T}, s) where {T <: Vec3D} = VecArray(x = similar(in_field, nonzero_eltype(T), s), y = similar(in_field, nonzero_eltype(T), s), z = similar(in_field, nonzero_eltype(T), s))

function (Vop::LinearVoronoiOperator)(in_field::AbstractArray, op::F = Base.identity) where {F <: Function}
    is_proper_size(in_field, n_input(Vop)) || throw(DimensionMismatch("Input array doesn't seem to be a $(name_input(Vop)) field"))
    s = construct_new_node_index(size(in_field)..., n_output(Vop))
    out_field = my_similar(in_field, Base.promote_op(*, eltype(eltype(Vop.weights)), Base.promote_op(op, eltype(in_field))), s)
    return Vop(out_field, in_field, op)
end

include("node_transformations.jl")
include("cell_vector_to_edge_transformations.jl")
include("cell_velocity_reconstruction.jl")
include("kinetic_energy_reconstruction.jl")
include("differential_operators.jl")
include("filtering_operators.jl")
include("tangential_velocity_reconstruction.jl")

end
