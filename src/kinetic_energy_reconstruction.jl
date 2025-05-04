abstract type KineticEnergyReconstruction <: NonLinearVoronoiOperator end
name_input(::KineticEnergyReconstruction) = "edge"

@inline square(x) = x * x

function (Vop::KineticEnergyReconstruction)(out_field::AbstractArray, in_field::AbstractArray)
    is_proper_size(in_field, n_input(Vop)) || throw(DimensionMismatch("Input array doesn't seem to be a $(name_input) field"))
    is_proper_size(out_field, n_output(Vop)) || throw(DimensionMismatch("Output array doesn't seem to be a $(name_output(Vop)) field"))

    weighted_sum_transformation!(out_field, in_field, Vop.weights, Vop.indices, square)

    return out_field
end

function (Vop::KineticEnergyReconstruction)(in_field::AbstractArray)
    is_proper_size(in_field, n_input(Vop)) || throw(DimensionMismatch("Input array doesn't seem to be a $(name_input(Vop)) field"))
    s = construct_new_node_index(size(in_field)..., n_output(Vop))
    out_field = my_similar(in_field, Base.promote_op(*, eltype(eltype(Vop.weights)), Base.promote_op(square, eltype(in_field))), s)
    return Vop(out_field, in_field)
end

function (Vop::KineticEnergyReconstruction)(out_field::AbstractArray, op::F, in_field::AbstractArray) where {F <: Function}
    is_proper_size(in_field, n_input(Vop)) || throw(DimensionMismatch("Input array doesn't seem to be a $(name_input(Vop)) field"))
    is_proper_size(out_field, n_output(Vop)) || throw(DimensionMismatch("Output array doesn't seem to be a $(name_output(Vop)) field"))

    weighted_sum_transformation!(out_field, op, in_field, Vop.weights, Vop.indices, square)

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

function compute_weights_vertex_kinetic_energy_modified!(w, aV, Lev, dc, edgesOnVertex)

    @inbounds for v in eachindex(edgesOnVertex)
        e1, e2, e3 = edgesOnVertex[v]
        term = inv(aV[v] * 2)
        Lev1, Lev2, Lev3 = Lev[v]

        w1 = term * Lev1 * dc[e1]
        w2 = term * Lev2 * dc[e2]
        w3 = term * Lev3 * dc[e3]

        w[v] = (w1, w2, w3)
    end
    return w
end

struct CellKineticEnergyMPAS{N_MAX, TI, TF} <: CellKineticEnergyReconstruction{N_MAX, TI, TF}
    vertexReconstruction::VertexKineticEnergyGassmann{TI, TF}
    RinglerReconstruction::CellKineticEnergyRingler{N_MAX, TI, TF}
    weightsVertexToCell::Vector{ImmutableVector{N_MAX, TF}}
    verticesOnCell::ImVecArray{N_MAX, TI, 1}
    alpha::TF
    kv1d::Base.RefValue{Vector{TF}}
    kv2d::Base.RefValue{Matrix{TF}}
    kv3d::Base.RefValue{Array{TF, 3}}
end

function compute_vertex_to_cell_weight!(w::AbstractVector{ImmutableVector{N_MAX, TF}}, verticesOnCell, areaCell, kiteAreaOnVertex, cellsOnVertex) where {N_MAX, TF}
    aux = Vector{TF}(undef, N_MAX)
    @inbounds for c in eachindex(areaCell)
        Ac = areaCell[c]
        fill!(aux, zero(TF))
        voc = verticesOnCell[c]
        l = length(voc)
        for i in Base.OneTo(l)
            v = voc[i]
            aux[i] = VoronoiMeshes.select_kite_area(kiteAreaOnVertex, cellsOnVertex, v, c) / Ac
        end
        w[c] = ImmutableVector{N_MAX}(ntuple(j -> getindex(aux, j), Val{N_MAX}()), l)
    end
    return w
end

function compute_vertex_to_cell_weight(mesh::AbstractVoronoiMesh)
    w = Vector{ImmutableVector{max_edges(typeof(mesh.cells)), eltype(mesh.cells.area)}}(undef, mesh.cells.n)
    return compute_vertex_to_cell_weight!(w, mesh.cells.vertices, mesh.cells.area, mesh.vertices.kiteAreas, mesh.vertices.cells)
end

function CellKineticEnergyMPAS(mesh::AbstractVoronoiMesh, alpha = 1 - 0.375)
    T = float_type(typeof(mesh.cells))
    return CellKineticEnergyMPAS(VertexKineticEnergyGassmann(mesh), CellKineticEnergyRingler(mesh), compute_vertex_to_cell_weight(mesh), mesh.cells.vertices, alpha, Ref{Vector{T}}(), Ref{Matrix{T}}(), Ref{Array{T, 3}}())
end

function get_proper_kv(ckm::CellKineticEnergyMPAS, ::Vector{TF}) where {TF}
    if !isassigned(ckm.kv1d)
        ckm.kv1d[] = Vector{TF}(undef, length(ckm.vertexReconstruction.indices))
    end
    return ckm.kv1d[]
end

function get_proper_kv(ckm::CellKineticEnergyMPAS, u::Matrix{TF}) where {TF}
    if !isassigned(ckm.kv2d)
        ckm.kv2d[] = Matrix{TF}(undef, size(u, 1), length(ckm.vertexReconstruction.indices))
    end
    return ckm.kv2d[]
end

function get_proper_kv(ckm::CellKineticEnergyMPAS, u::Array{TF, 3}) where {TF}
    if !isassigned(ckm.kv3d)
        ckm.kv3d[] = Array{TF, 3}(undef, size(u, 1), length(ckm.vertexReconstruction.indices), size(u, 3))
    end
    return ckm.kv3d[]
end

struct Combination{T} <: Function
    a::T
end

@inline (L::Combination)(x, y) = muladd(L.a, (x - y), y)

function (ckm::CellKineticEnergyMPAS)(c_field::AbstractArray, u::AbstractArray)
    is_proper_size(c_field, length(ckm.weightsVertexToCell)) || throw(DimensionMismatch("Output array doesn't seem to be a cell field"))
    is_proper_size(u, ckm.RinglerReconstruction.n) || throw(DimensionMismatch("Input array doesn't seem to be an edge field"))

    kv = get_proper_kv(ckm, u)
    ckm.vertexReconstruction(kv, u)
    ckm.RinglerReconstruction(c_field, u)

    f = Combination(ckm.alpha)

    weighted_sum_transformation!(c_field, f, kv, ckm.weightsVertexToCell, ckm.verticesOnCell)

    return c_field
end

function (kc::CellKineticEnergyMPAS)(e_field::AbstractArray)
    is_proper_size(e_field, kc.RinglerReconstruction.n) || throw(DimensionMismatch("Input array doesn't seem to be an edge field"))
    s = construct_new_node_index(size(e_field)..., length(kc.RinglerReconstruction.indices))
    c_field = my_similar(e_field, Base.promote_op(*, eltype(eltype(kc.RinglerReconstruction.weights)), eltype(e_field)), s)
    return kc(c_field, e_field)
end

struct OpAtimes{F, T} <: Function
    op::F
    a::T
end

@inline (O::OpAtimes)(x, y) = O.op(x, O.a * y)

function (ckm::CellKineticEnergyMPAS)(c_field::AbstractArray, op::F, u::AbstractArray) where {F <: Union{typeof(Base.:+), typeof(Base.:-)}}
    is_proper_size(c_field, length(ckm.weightsVertexToCell)) || throw(DimensionMismatch("Output array doesn't seem to be a cell field"))
    is_proper_size(u, ckm.RinglerReconstruction.n) || throw(DimensionMismatch("Input array doesn't seem to be an edge field"))

    kv = get_proper_kv(ckm, u)
    ckm.vertexReconstruction(kv, u)

    f1 = OpAtimes(op, ckm.alpha)

    ckm.RinglerReconstruction(c_field, f1, u)

    f2 = OpAtimes(op, 1.0 - ckm.alpha)

    weighted_sum_transformation!(c_field, f2, kv, ckm.weightsVertexToCell, ckm.verticesOnCell)

    return c_field
end

struct CellKineticEnergyVelRecon{N_MAX, TI, TF, TR <: CellVelocityReconstruction{N_MAX, TI, TF}} <: CellKineticEnergyReconstruction{N_MAX, TI, TF}
    uR::TR
end

CellKineticEnergyPerot(mesh::AbstractVoronoiMesh) = CellKineticEnergyVelRecon(CellVelocityReconstructionPerot(mesh))

@inline kinetic_energy(x) = 0.5 * (x ⋅ x)
@inline kinetic_energy(y, x) = 0.5 * (x ⋅ x)

function (kc::CellKineticEnergyVelRecon)(c_field::AbstractArray, e_field::AbstractArray)
    is_proper_size(e_field, kc.uR.n) || throw(DimensionMismatch("Input array doesn't seem to be an edge field"))
    is_proper_size(c_field, length(kc.uR.indices)) || throw(DimensionMismatch("Output array doesn't seem to be a cell field"))

    kc.uR(c_field, kinetic_energy, e_field)

    return c_field
end

function (kc::CellKineticEnergyVelRecon)(e_field::AbstractArray)
    is_proper_size(e_field, kc.uR.n) || throw(DimensionMismatch("Input array doesn't seem to be an edge field"))
    s = construct_new_node_index(size(e_field)..., length(kc.uR.indices))
    c_field = my_similar(e_field, Base.promote_op(dot, eltype(eltype(kc.uR.weights)), eltype(eltype(kc.uR.weights))), s)
    return kc(c_field, e_field)
end

struct OpKineticEnergy{F} <: Function
    op::F
end

@inline (O::OpKineticEnergy)(x, y) = O.op(x, kinetic_energy(y))

function (kc::CellKineticEnergyVelRecon)(c_field::AbstractArray, op::F, e_field::AbstractArray) where {F <: Function}
    is_proper_size(e_field, kc.uR.n) || throw(DimensionMismatch("Input array doesn't seem to be an edge field"))
    is_proper_size(c_field, length(kc.uR.indices)) || throw(DimensionMismatch("Output array doesn't seem to be a cell field"))

    f = OpKineticEnergy(op)
    kc.uR(c_field, f, e_field)

    return c_field
end

struct VertexKineticEnergyVelRecon{TI, TF, TR <: VertexVelocityReconstruction{TI, TF}} <: VertexKineticEnergyReconstruction{TI, TF}
    uR::TR
end

VertexKineticEnergyPerot(mesh::AbstractVoronoiMesh) = VertexKineticEnergyVelRecon(VertexVelocityReconstructionPerot(mesh))

function (kv::VertexKineticEnergyVelRecon)(v_field::AbstractArray, e_field::AbstractArray)
    is_proper_size(e_field, kv.uR.n) || throw(DimensionMismatch("Input array doesn't seem to be an edge field"))
    is_proper_size(v_field, length(kv.uR.indices)) || throw(DimensionMismatch("Output array doesn't seem to be a vertex field"))

    kv.uR(v_field, kinetic_energy, e_field)

    return v_field
end

function (kv::VertexKineticEnergyVelRecon)(e_field::AbstractArray)
    is_proper_size(e_field, kv.uR.n) || throw(DimensionMismatch("Input array doesn't seem to be an edge field"))
    s = construct_new_node_index(size(e_field)..., length(kv.uR.indices))
    v_field = my_similar(e_field, Base.promote_op(dot, eltype(eltype(kv.uR.weights)), eltype(eltype(kv.uR.weights))), s)
    return kv(v_field, e_field)
end

function (kv::VertexKineticEnergyVelRecon)(v_field::AbstractArray, op::F, e_field::AbstractArray) where {F <: Function}
    is_proper_size(e_field, kv.uR.n) || throw(DimensionMismatch("Input array doesn't seem to be an edge field"))
    is_proper_size(v_field, length(kv.uR.indices)) || throw(DimensionMismatch("Output array doesn't seem to be a vertex field"))

    f = OpKineticEnergy(op)
    kv.uR(v_field, f, e_field)

    return v_field
end

