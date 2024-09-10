abstract type TangentialVelocityReconstruction{N_MAX, TF, TI} <: LinearVoronoiOperator end
name_input(::TangentialVelocityReconstruction) = "edge"
name_output(::TangentialVelocityReconstruction) = "edge"
n_input(a::TangentialVelocityReconstruction) = n_output(a)

struct TangentialVelocityReconstructionThuburn{N_MAX, TF, TI} <: TangentialVelocityReconstruction{N_MAX, TF, TI}
    indices::Vector{ImmutableVector{N_MAX, TI}}
    weights::Vector{ImmutableVector{N_MAX, TF}}
end

function TangentialVelocityReconstructionThuburn(mesh::VoronoiMesh)
    return TangentialVelocityReconstructionThuburn(mesh.edgesOnEdge, mesh.weightsOnEdge)
end
