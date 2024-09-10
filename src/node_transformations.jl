abstract type VertexToEdgeTransformation <: LinearVoronoiOperator end

name_input(::VertexToEdgeTransformation) = "vertex"
name_output(::VertexToEdgeTransformation) = "edge"


struct VertexToEdgeMean{TI} <: VertexToEdgeTransformation
    n::Int
    indices::Vector{NTuple{2, TI}}
end

VertexToEdgeMean(vertices::Union{<:VertexBase, <:VertexInfo}, edges::Union{<:EdgeBase, <:EdgeInfo}) = VertexToEdgeMean(vertices.n, edges.indices.vertices)
VertexToEdgeMean(mesh::VoronoiMesh) = VertexToEdgeMean(mesh.vertices.base, mesh.edges.base)

@inbounds @inline (v2e::VertexToEdgeMean)(v_field::AbstractArray{<:Any, N}, inds::Vararg{T, N}) where {N, T <: Integer} = to_mean_transformation(v_field, inds, v2e.indices)

function (v2e::VertexToEdgeMean)(e_field::AbstractArray, v_field::AbstractArray)
    is_proper_size(v_field, v2e.n) || throw(DomainError(v_field, "Input array doesn't seem to be a vertex field"))
    is_proper_size(e_field, length(v2e.indices)) || throw(DomainError(e_field, "Output array doesn't seem to be an edge field"))

    to_mean_transformation!(e_field, v_field, v2e.indices)

    return e_field
end

function (v2e::VertexToEdgeMean)(v_field::AbstractArray)
    is_proper_size(v_field, v2e.n) || throw(DomainError(v_field, "Input array doesn't seem to be a vertex field"))
    s = construct_new_node_index(size(v_field)..., length(v2e.indices))
    e_field = similar(v_field, s)
    return v2e(e_field, v_field)
end

function (v2e::VertexToEdgeMean)(e_field::AbstractArray, op::F, v_field::AbstractArray) where {F <: Function}
    is_proper_size(v_field, v2e.n) || throw(DomainError(v_field, "Input array doesn't seem to be a vertex field"))
    is_proper_size(e_field, length(v2e.indices)) || throw(DomainError(e_field, "Output array doesn't seem to be an edge field"))

    to_mean_transformation!(e_field, op, v_field, v2e.indices)

    return e_field
end

struct VertexToEdgeWeighted{TI, TF} <: VertexToEdgeTransformation
    n::Int
    indices::Vector{NTuple{2, TI}}
    weights::Vector{NTuple{2, TF}}
end

struct VertexToEdgeInterpolation{TI, TF} <: VertexToEdgeTransformation
    base::VertexToEdgeWeighted{TI, TF}
    VertexToEdgeInterpolation(base::VertexToEdgeWeighted{TI, TF}) where {TI, TF} = new{TI, TF}(base)
end

function Base.getproperty(v2e::VertexToEdgeInterpolation, s::Symbol)
    if s === :n
        return (getfield(v2e, :base)).n
    elseif s === :indices
        return (getfield(v2e, :base)).indices
    elseif s === :weights
        return (getfield(v2e, :base)).weights
    else
        return getfield(v2e, s)
    end
end

function compute_interpolation_weights_vertex_to_edge_periodic(epos, vpos, voe, dvEdge, xp::Number, yp::Number)
    weights = Vector{NTuple{2, eltype(dvEdge)}}(undef, length(epos))
    @inbounds for e in eachindex(voe)
        v1, v2 = voe[e]
        dv = dvEdge[e]
        ep = epos[e]
        l_e_v1 = norm(closest(ep, vpos[v1], xp, yp) - ep)
        #w1 = (dv - l_e_v1)/dv
        w1 = 1 - l_e_v1 / dv
        w2 = 1 - w1
        weights[e] = (w1, w2)
    end
    return weights
end

function compute_interpolation_weights_vertex_to_edge(vertices::Union{<:VertexBase{false}, <:VertexInfo{false}}, edges::EdgeInfo{false}, xp::Number, yp::Number)
    return compute_interpolation_weights_vertex_to_edge_periodic(edges.position, vertices.position, edges.indices.vertices, edges.dv, xp, yp)
end

function VertexToEdgeInterpolation(vertices::Union{<:VertexBase{false}, <:VertexInfo{false}}, edges::EdgeInfo{false}, xp::Number, yp::Number)
    weights = compute_interpolation_weights_vertex_to_edge(vertices, edges, xp, yp)
    return VertexToEdgeInterpolation(VertexToEdgeWeighted(vertices.n, edges.indices.vertices, weights))
end

VertexToEdgeInterpolation(mesh::VoronoiMesh{false}) = VertexToEdgeInterpolation(mesh.vertices.base, mesh.edges, mesh.attributes[:x_period]::Float64, mesh.attributes[:y_period]::Float64)

struct VertexToEdgePiecewise{TI, TF} <: VertexToEdgeTransformation
    base::VertexToEdgeWeighted{TI, TF}
    VertexToEdgePiecewise(base::VertexToEdgeWeighted{TI, TF}) where {TI, TF} = new{TI, TF}(base)
end

function Base.getproperty(v2e::VertexToEdgePiecewise, s::Symbol)
    if s === :n
        return (getfield(v2e, :base)).n
    elseif s === :indices
        return (getfield(v2e, :base)).indices
    elseif s === :weights
        return (getfield(v2e, :base)).weights
    else
        return getfield(v2e, s)
    end
end

function VertexToEdgePiecewise(vertices::Union{<:VertexBase{false}, <:VertexInfo{false}}, edges::EdgeInfo{false}, xp::Number, yp::Number)
    weights = compute_interpolation_weights_vertex_to_edge(vertices, edges, xp, yp)
    weights .= reverse.(weights)
    return VertexToEdgePiecewise(VertexToEdgeWeighted(vertices.n, edges.indices.vertices, weights))
end

VertexToEdgePiecewise(mesh::VoronoiMesh{false}) = VertexToEdgePiecewise(mesh.vertices.base, mesh.edges, mesh.attributes[:x_period]::Float64, mesh.attributes[:y_period]::Float64)

struct VertexToEdgeArea{TI, TF} <: VertexToEdgeTransformation
    base::VertexToEdgeWeighted{TI, TF}
    VertexToEdgeArea(base::VertexToEdgeWeighted{TI, TF}) where {TI, TF} = new{TI, TF}(base)
end

function Base.getproperty(v2e::VertexToEdgeArea, s::Symbol)
    if s === :n
        return (getfield(v2e, :base)).n
    elseif s === :indices
        return (getfield(v2e, :base)).indices
    elseif s === :weights
        return (getfield(v2e, :base)).weights
    else
        return getfield(v2e, s)
    end
end

function compute_area_weights_vertex_to_edge(voe, areaTriangles)
    weights = Vector{NTuple{2, eltype(areaTriangles)}}(undef, length(voe))
    @inbounds for e in eachindex(voe)
        v1, v2 = voe[e]
        a1 = areaTriangles[v1]
        a2 = areaTriangles[v2]
        at = a1 + a2
        w1 = a1 / at
        w2 = 1 - w1
        weights[e] = (w1, w2)
    end
    return weights
end

function VertexToEdgeArea(vertices::VertexInfo, edges::Union{<:EdgeInfo, <:EdgeBase})
    weights = compute_area_weights_vertex_to_edge(edges.indices.vertices, vertices.area)
    return VertexToEdgeArea(VertexToEdgeWeighted(vertices.n, edges.indices.vertices, weights))
end

VertexToEdgeArea(mesh::VoronoiMesh) = VertexToEdgeArea(mesh.vertices, mesh.edges.base)

abstract type CellToEdgeTransformation <: LinearVoronoiOperator end

name_input(::CellToEdgeTransformation) = "cell"
name_output(::CellToEdgeTransformation) = "edge"

struct CellToEdgeMean{TI} <: CellToEdgeTransformation
    n::Int
    indices::Vector{NTuple{2, TI}}
end

CellToEdgeMean(cells::Union{<:CellBase, <:CellInfo}, edges::Union{<:EdgeBase, <:EdgeInfo}) = CellToEdgeMean(cells.n, edges.indices.cells)
CellToEdgeMean(mesh::VoronoiMesh) = CellToEdgeMean(mesh.cells.base, mesh.edges.base)

@inbounds @inline (c2e::CellToEdgeMean)(c_field::AbstractArray{<:Any, N}, inds::Vararg{T, N}) where {N, T <: Integer} = to_mean_transformation(c_field, inds, c2e.indices)

function (c2e::CellToEdgeMean)(e_field::AbstractArray, c_field::AbstractArray)
    is_proper_size(c_field, c2e.n) || throw(DomainError(c_field, "Input array doesn't seem to be a cell field"))
    is_proper_size(e_field, length(c2e.indices)) || throw(DomainError(e_field, "Output array doesn't seem to be an edge field"))

    to_mean_transformation!(e_field, c_field, c2e.indices)

    return e_field
end

function (c2e::CellToEdgeMean)(c_field::AbstractArray)
    is_proper_size(c_field, c2e.n) || throw(DomainError(c_field, "Input array doesn't seem to be a cell field"))
    s = construct_new_node_index(size(c_field)..., length(c2e.indices))
    e_field = similar(c_field, s)
    return c2e(e_field, c_field)
end

function (c2e::CellToEdgeMean)(e_field::AbstractArray, op::F, c_field::AbstractArray) where {F <: Function}
    is_proper_size(c_field, c2e.n) || throw(DomainError(c_field, "Input array doesn't seem to be a cell field"))
    is_proper_size(e_field, length(c2e.indices)) || throw(DomainError(e_field, "Output array doesn't seem to be an edge field"))

    to_mean_transformation!(e_field, op, c_field, c2e.indices)

    return e_field
end

struct CellToEdgeBaricentric{TI, TF} <: CellToEdgeTransformation
    n::Int
    indices::Vector{NTuple{3, TI}}
    weights::Vector{NTuple{3, TF}}
end

function compute_baricentric_cell_to_edge_periodic!(w, inds, edge_pos, cell_pos, v_pos, areaTriangle, verticesOnEdge, cellsOnVertex, x_period::Number, y_period::Number)

    @inbounds for e in eachindex(verticesOnEdge)
        e_pos = edge_pos[e]
        v1, v2 = verticesOnEdge[e]
        v1_pos = closest(e_pos, v_pos[v1], x_period, y_period)
        v2_pos = closest(e_pos, v_pos[v2], x_period, y_period)

        e_mid = 0.5 * (v1_pos + v2_pos)

        c11, c12, c13 = cellsOnVertex[v1]
        c11_pos = closest(e_pos, cell_pos[c11], x_period, y_period)
        c12_pos = closest(e_pos, cell_pos[c12], x_period, y_period)
        c13_pos = closest(e_pos, cell_pos[c13], x_period, y_period)

        at = areaTriangle[v1]

        in_here, w1, w2, w3, _ = in_triangle(e_mid, at, c11_pos, c12_pos, c13_pos)
        if in_here
            i1 = c11
            i2 = c12
            i3 = c13
        else
            c21, c22, c23 = cellsOnVertex[v2]
            c21_pos = closest(e_pos, cell_pos[c21], x_period, y_period)
            c22_pos = closest(e_pos, cell_pos[c22], x_period, y_period)
            c23_pos = closest(e_pos, cell_pos[c23], x_period, y_period)

            at = areaTriangle[v2]
            in_here, w1, w2, w3, _ = in_triangle(e_mid, at, c21_pos, c22_pos, c23_pos)
            i1 = c21
            i2 = c22
            i3 = c23
        end
        inv_at = inv(at)
        w1 *= inv_at
        #w2 *= inv_at
        w3 *= inv_at

        #make sure it most likely sums to one (might not due to float point precision)
        w2 = 1 - (w1 + w3)

        w[e] = (w1, w2, w3)
        inds[e] = map(Int, (i1, i2, i3))
    end

    return w, inds
end

function compute_baricentric_cell_to_edge_periodic(edge_pos, cell_pos, vertex_pos, areaTriangle, verticesOnEdge, cellsOnVertex, x_period::Number, y_period::Number)
    TF = eltype(cell_pos.x)
    w = Vector{NTuple{3, TF}}(undef, length(verticesOnEdge))
    inds = Vector{NTuple{3, Int}}(undef, length(verticesOnEdge))
    return compute_baricentric_cell_to_edge_periodic!(w, inds, edge_pos, cell_pos, vertex_pos, areaTriangle, verticesOnEdge, cellsOnVertex, x_period, y_period)
end

function compute_baricentric_cell_to_edge_periodic(m::VoronoiMesh)
    compute_baricentric_cell_to_edge_periodic(
        m.edges.position, m.cells.position, m.vertices.position, m.vertices.area,
        m.edges.indices.vertices, m.vertices.indices.cells,
        m.attributes[:x_period]::Float64, m.attributes[:y_period]::Float64
    )
end

function CellToEdgeBaricentric(m::VoronoiMesh{false})
    weights, cellsOnEdgeBaricentric = compute_baricentric_cell_to_edge_periodic(m)
    ncells = m.nCells
    return CellToEdgeBaricentric(ncells, cellsOnEdgeBaricentric, weights)
end
