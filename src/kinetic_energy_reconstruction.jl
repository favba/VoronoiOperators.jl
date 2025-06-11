abstract type KineticEnergyReconstruction <: NonLinearVoronoiOperator end

@inline square(x) = x * x
name_input(::KineticEnergyReconstruction) = "edge"
n_input(o::KineticEnergyReconstruction) = o.n
n_output(o::KineticEnergyReconstruction) = length(o.indices)
out_eltype(Vop::KineticEnergyReconstruction, in_field, op::F = Base.identity) where {F} = Base.promote_op(*, eltype(eltype(Vop.weights)), Base.promote_op(square∘op, eltype(in_field)))

function (Vop::KineticEnergyReconstruction)(out_field::AbstractArray, in_field::AbstractArray, op::F = Base.identity) where {F}
    is_proper_size(in_field, n_input(Vop)) || throw(DimensionMismatch("Input array doesn't seem to be a $(name_input) field"))
    is_proper_size(out_field, n_output(Vop)) || throw(DimensionMismatch("Output array doesn't seem to be a $(name_output(Vop)) field"))

    weighted_sum_transformation!(out_field, in_field, Vop.weights, Vop.indices, square∘op)

    return out_field
end

function (Vop::KineticEnergyReconstruction)(in_field::AbstractArray, op::F = Base.identity) where {F}
    is_proper_size(in_field, n_input(Vop)) || throw(DimensionMismatch("Input array doesn't seem to be a $(name_input(Vop)) field"))
    s = construct_new_node_index(size(in_field)..., n_output(Vop))
    out_field = my_similar(in_field, out_eltype(Vop, in_field, op), s)
    return Vop(out_field, in_field, op)
end

function (Vop::KineticEnergyReconstruction)(out_field::AbstractArray, op::F, in_field::AbstractArray, op2::F2 = Base.identity) where {F <: Function, F2}
    is_proper_size(in_field, n_input(Vop)) || throw(DimensionMismatch("Input array doesn't seem to be a $(name_input(Vop)) field"))
    is_proper_size(out_field, n_output(Vop)) || throw(DimensionMismatch("Output array doesn't seem to be a $(name_output(Vop)) field"))

    weighted_sum_transformation!(out_field, op, in_field, Vop.weights, Vop.indices, square∘op2)

    return out_field
end

abstract type CellKineticEnergyReconstruction{N_MAX, TI, TF} <: KineticEnergyReconstruction end
name_output(::CellKineticEnergyReconstruction) = "cell"

struct CellKineticEnergyRingler{N_MAX, TI, TF} <: CellKineticEnergyReconstruction{N_MAX, TI, TF}
    n::Int
    indices::ImVecArray{N_MAX, TI, 1}
    weights::ImVecArray{N_MAX, TF, 1}
end

function CellKineticEnergyRingler(cells, edges)
    w = compute_weights_edge_to_cell_ringler(cells, edges)
    return CellKineticEnergyRingler(edges.n, cells.edges, w)
end

CellKineticEnergyRingler(m::AbstractVoronoiMesh) = CellKineticEnergyRingler(m.cells, m.edges)

abstract type VertexKineticEnergyReconstruction{TI, TF} <: KineticEnergyReconstruction end
name_output(::VertexKineticEnergyReconstruction) = "vertex"

struct VertexKineticEnergyGassmann{TI, TF} <: VertexKineticEnergyReconstruction{TI, TF}
    n::Int
    indices::Vector{NTuple{3, TI}}
    weights::Vector{NTuple{3, TF}}
end

function compute_weights_vertex_kinetic_energy!(w, aV, Le, dc, edgesOnVertex)

    @inbounds for v in eachindex(edgesOnVertex)
        e1, e2, e3 = edgesOnVertex[v]
        term = inv(aV[v] * 4)

        w1 = term * Le[e1] * dc[e1]
        w2 = term * Le[e2] * dc[e2]
        w3 = term * Le[e3] * dc[e3]

        w[v] = (w1, w2, w3)
    end
    return w
end

function compute_weights_vertex_kinetic_energy(aV, Le, dc, edgesOnVertex)
    w = Vector{NTuple{3, eltype(aV)}}(undef, length(edgesOnVertex))
    return compute_weights_vertex_kinetic_energy!(w, aV, Le, dc, edgesOnVertex)
end

function VertexKineticEnergyGassmann(vertices, edges)
    w = compute_weights_vertex_kinetic_energy(vertices.area, edges.length, edges.lengthDual, vertices.edges)
    return VertexKineticEnergyGassmann(edges.n, vertices.edges, w)
end

VertexKineticEnergyGassmann(m::AbstractVoronoiMesh) = VertexKineticEnergyGassmann(m.vertices, m.edges)

struct CellKineticEnergyVertexWeighted{N_MAX, TI, TF,
                                       TVR<:VertexKineticEnergyReconstruction{TI, TF},
                                       TCR<:CellKineticEnergyReconstruction{N_MAX, TI, TF},
                                       TVC<:VertexToCellTransformation{N_MAX, TI, TF}} <: CellKineticEnergyReconstruction{N_MAX, TI, TF}
    vertexReconstruction::TVR
    cellReconstruction::TCR
    vertexToCell::TVC
    alpha::TF
    kv1d::Base.RefValue{Vector{TF}}
    kv2d::Base.RefValue{Matrix{TF}}
    kv3d::Base.RefValue{Array{TF, 3}}
end

n_input(o::CellKineticEnergyVertexWeighted) = n_input(o.cellReconstruction)
n_output(o::CellKineticEnergyVertexWeighted) = n_output(o.cellReconstruction)
out_eltype(o::CellKineticEnergyVertexWeighted, in_field, op::F = Base.identity) where {F} = out_eltype(o.cellReconstruction, in_field, op)

function CellKineticEnergyVertexWeighted(vertexReconstruction::VertexKineticEnergyReconstruction{TI,TF}, cellReconstruction, vertexToCell, alpha = 1 - 0.375) where {TI, TF}
    return CellKineticEnergyVertexWeighted(vertexReconstruction, cellReconstruction, vertexToCell, alpha, Ref{Vector{TF}}(), Ref{Matrix{TF}}(), Ref{Array{TF, 3}}())
end

function get_proper_kv(ckm::CellKineticEnergyVertexWeighted, ::Vector{TF}) where {TF}
    if !isassigned(ckm.kv1d)
        ckm.kv1d[] = Vector{TF}(undef, n_output(ckm.vertexReconstruction))
    end
    return ckm.kv1d[]
end

function get_proper_kv(ckm::CellKineticEnergyVertexWeighted, u::Matrix{TF}) where {TF}
    if !isassigned(ckm.kv2d)
        ckm.kv2d[] = Matrix{TF}(undef, size(u, 1), n_output(ckm.vertexReconstruction))
    end
    return ckm.kv2d[]
end

function get_proper_kv(ckm::CellKineticEnergyVertexWeighted, u::Array{TF, 3}) where {TF}
    if !isassigned(ckm.kv3d)
        ckm.kv3d[] = Array{TF, 3}(undef, size(u, 1), n_output(ckm.vertexReconstruction), size(u, 3))
    end
    return ckm.kv3d[]
end

struct Combination{T} <: Function
    a::T
end

@inline (L::Combination)(x, y) = muladd(L.a, (x - y), y)

function (ckm::CellKineticEnergyVertexWeighted)(c_field::AbstractArray, kv::AbstractArray, u::AbstractArray, op::F = Base.identity) where F
    is_proper_size(c_field, n_output(ckm)) || throw(DimensionMismatch("Output array doesn't seem to be a cell field"))
    is_proper_size(kv, n_output(ckm.vertexReconstruction)) || throw(DimensionMismatch("Intermediary array doesn't seem to be a vertex field"))
    is_proper_size(u, n_input(ckm)) || throw(DimensionMismatch("Input array doesn't seem to be an edge field"))

    ckm.vertexReconstruction(kv, u, op)
    ckm.cellReconstruction(c_field, u, op)

    f = Combination(ckm.alpha)

    ckm.vertexToCell(c_field, f, kv)

    return c_field
end

(ckm::CellKineticEnergyVertexWeighted)(c_field::AbstractArray, u::AbstractArray, op::F = Base.identity) where F = ckm(c_field, get_proper_kv(ckm,u), u, op)

function (kc::CellKineticEnergyVertexWeighted)(e_field::AbstractArray, op::F = Base.identity) where {F}
    is_proper_size(e_field, n_input(kc)) || throw(DimensionMismatch("Input array doesn't seem to be an edge field"))
    s = construct_new_node_index(size(e_field)..., n_output(kc))
    c_field = my_similar(e_field, out_eltype(kc.cellReconstruction, e_field, op), s)
    return kc(c_field, e_field, op)
end

struct OpAtimes{F, T} <: Function
    op::F
    a::T
end

@inline (O::OpAtimes)(x, y) = O.op(x, O.a * y)

function (ckm::CellKineticEnergyVertexWeighted)(c_field::AbstractArray, kv::AbstractArray, op::F, u::AbstractArray, op2::F2 = Base.identity) where {F <: Union{typeof(Base.:+), typeof(Base.:-)}, F2}
    is_proper_size(c_field, n_output(ckm)) || throw(DimensionMismatch("Output array doesn't seem to be a cell field"))
    is_proper_size(kv, n_output(ckm.vertexReconstruction)) || throw(DimensionMismatch("Intermediary array doesn't seem to be a vertex field"))
    is_proper_size(u, n_input(ckm)) || throw(DimensionMismatch("Input array doesn't seem to be an edge field"))

    ckm.vertexReconstruction(kv, u, op2)

    f1 = OpAtimes(op, ckm.alpha)

    ckm.cellReconstruction(c_field, f1, u, op2)

    f2 = OpAtimes(op, 1.0 - ckm.alpha)

    ckm.vertexToCell(c_field, f2, kv)

    return c_field
end

(ckm::CellKineticEnergyVertexWeighted)(c_field::AbstractArray, op::F, u::AbstractArray, op2::F2 = Base.identity) where {F <: Union{typeof(Base.:+), typeof(Base.:-)}, F2} =
    ckm(c_field, get_proper_kv(ckm, u), op, u, op2)

const CellKineticEnergyMPAS{N_MAX, TI, TF} = CellKineticEnergyVertexWeighted{N_MAX, TI, TF,
                                                                             VertexKineticEnergyGassmann{TI, TF},
                                                                             CellKineticEnergyRingler{N_MAX, TI, TF},
                                                                             VertexToCellArea{N_MAX, TI, TF}}

function CellKineticEnergyMPAS(mesh::AbstractVoronoiMesh, alpha = 1 - 0.375)
    T = float_type(typeof(mesh.cells))
    return CellKineticEnergyVertexWeighted(VertexKineticEnergyGassmann(mesh), CellKineticEnergyRingler(mesh), VertexToCellArea(mesh), alpha, Ref{Vector{T}}(), Ref{Matrix{T}}(), Ref{Array{T, 3}}())
end

struct CellKineticEnergyVelRecon{N_MAX, TI, TF, TR <: CellVelocityReconstruction{N_MAX, TI, TF}} <: CellKineticEnergyReconstruction{N_MAX, TI, TF}
    uR::TR
end

struct VertexKineticEnergyVelRecon{TI, TF, TR <: VertexVelocityReconstruction{TI, TF}} <: VertexKineticEnergyReconstruction{TI, TF}
    uR::TR
end

const KineticEnergyVelRecon = Union{<:CellKineticEnergyVelRecon, <:VertexKineticEnergyVelRecon}

@inline kinetic_energy(x) = 0.5 * (x ⋅ x) 
@inline kinetic_energy(::Any, x) = 0.5 * (x ⋅ x)

n_input(o::KineticEnergyVelRecon) = n_input(o.uR)
n_output(o::KineticEnergyVelRecon) = n_output(o.uR)
out_eltype(Vop::KineticEnergyVelRecon, in_field, op::F = Base.identity) where {F} = Base.promote_op(kinetic_energy, out_eltype(Vop.uR, in_field, op))

function (kc::KineticEnergyVelRecon)(c_field::AbstractArray, e_field::AbstractArray, op::F = Base.identity) where {F}
    is_proper_size(e_field, n_input(kc)) || throw(DimensionMismatch("Input array doesn't seem to be an edge field"))
    is_proper_size(c_field, n_output(kc)) || throw(DimensionMismatch("Output array doesn't seem to be a $(name_output(kc)) field"))

    kc.uR(c_field, kinetic_energy, e_field, op)

    return c_field
end

function (kc::KineticEnergyVelRecon)(e_field::AbstractArray, op::F = Base.identity) where {F}
    is_proper_size(e_field, n_input(kc)) || throw(DimensionMismatch("Input array doesn't seem to be an edge field"))
    s = construct_new_node_index(size(e_field)..., n_output(kc))
    c_field = my_similar(e_field, out_eltype(kc, e_field, op), s)
    return kc(c_field, e_field)
end

struct OpKineticEnergy{F} <: Function
    op::F
end

@inline (O::OpKineticEnergy)(x, y) = O.op(x, kinetic_energy(y))

function (kc::KineticEnergyVelRecon)(c_field::AbstractArray, op::F, e_field::AbstractArray, op2::F2 = Base.identity) where {F <: Function, F2}
    is_proper_size(e_field, n_input(kc)) || throw(DimensionMismatch("Input array doesn't seem to be an edge field"))
    is_proper_size(c_field, n_output(kc)) || throw(DimensionMismatch("Output array doesn't seem to be a $(name_output(kc)) field"))

    f = OpKineticEnergy(op)
    kc.uR(c_field, f, e_field, op2)

    return c_field
end

const CellKineticEnergyPerot{N_MAX, TI, TF, TZ} = CellKineticEnergyVelRecon{N_MAX, TI, TF, CellVelocityReconstructionPerot{N_MAX, TI, TF, TZ}}

CellKineticEnergyPerot(mesh::AbstractVoronoiMesh) = CellKineticEnergyVelRecon(CellVelocityReconstructionPerot(mesh))

const VertexKineticEnergyPerot{TI, TF, TZ} = VertexKineticEnergyVelRecon{TI, TF, VertexVelocityReconstructionPerot{TI, TF, TZ}}

VertexKineticEnergyPerot(mesh::AbstractVoronoiMesh) = VertexKineticEnergyVelRecon(VertexVelocityReconstructionPerot(mesh))

const CellKineticEnergyPerotWeighted{N_MAX, TI, TF, TZ} = CellKineticEnergyVertexWeighted{N_MAX, TI, TF,
                                                                             VertexKineticEnergyPerot{TI, TF, TZ},
                                                                             CellKineticEnergyPerot{N_MAX, TI, TF, TZ},
                                                                             VertexToCellArea{N_MAX, TI, TF}}

function CellKineticEnergyPerotWeighted(mesh::AbstractVoronoiMesh, alpha = 1 - 0.375)
    T = float_type(typeof(mesh.cells))
    return CellKineticEnergyVertexWeighted(VertexKineticEnergyPerot(mesh), CellKineticEnergyPerot(mesh), VertexToCellArea(mesh), alpha, Ref{Vector{T}}(), Ref{Matrix{T}}(), Ref{Array{T, 3}}())
end
