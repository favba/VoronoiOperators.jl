abstract type VertexToEdgeTransformation <: LinearVoronoiOperator end

name_input(::VertexToEdgeTransformation) = "vertex"
name_output(::VertexToEdgeTransformation) = "edge"

struct VertexToEdgeMean{TI} <: VertexToEdgeTransformation
    n::Int
    indices::Vector{NTuple{2, TI}}
end

out_eltype(::VertexToEdgeMean, in_file::AbstractArray{T}, op::F = Base.identity) where {T, F <: Function} = Base.promote_op(/, Base.promote_op(op, T), Int)

transformation_function!(out_field::AbstractArray, in_field::AbstractArray, Vop::VertexToEdgeMean, op::F) where {F <: Function} = to_mean_transformation!(out_field, in_field, Vop.indices, op)

transformation_function!(out_field::AbstractArray, opt_out::F, in_field::AbstractArray, Vop::VertexToEdgeMean, op::F2) where {F <: Function, F2 <: Function} = to_mean_transformation!(out_field, opt_out, in_field, Vop.indices, op)

VertexToEdgeMean(vertices::Union{<:VertexBase, <:VertexInfo}, edges::Union{<:EdgeBase, <:EdgeInfo}) = VertexToEdgeMean(vertices.n, edges.indices.vertices)
VertexToEdgeMean(mesh::VoronoiMesh) = VertexToEdgeMean(mesh.vertices.base, mesh.edges.base)

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
    @parallel for e in eachindex(voe)
        @inbounds begin
        v1, v2 = voe[e]
        dv = dvEdge[e]
        ep = epos[e]
        l_e_v1 = norm(closest(ep, vpos[v1], xp, yp) - ep)
        #w1 = (dv - l_e_v1)/dv
        w1 = 1 - l_e_v1 / dv
        w2 = 1 - w1
        weights[e] = (w1, w2)
        end # inbounds
    end
    return weights
end

function compute_interpolation_weights_vertex_to_edge(epos, vpos, voe, dvEdge)
    R = norm(epos[1])
    weights = Vector{NTuple{2, eltype(dvEdge)}}(undef, length(epos))
    @parallel for e in eachindex(voe)
        @inbounds begin
        v1, _ = voe[e]
        dv = dvEdge[e]
        ep = epos[e]
        l_e_v1 = arc_length(R, vpos[v1], ep)
        w1 = 1 - l_e_v1 / dv
        w2 = 1 - w1
        weights[e] = (w1, w2)
        end # inbounds
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

function compute_interpolation_weights_vertex_to_edge(vertices::Union{<:VertexBase{true}, <:VertexInfo{true}}, edges::EdgeInfo{true})
    return compute_interpolation_weights_vertex_to_edge(edges.position, vertices.position, edges.indices.vertices, edges.dv)
end

function VertexToEdgeInterpolation(vertices::Union{<:VertexBase{true}, <:VertexInfo{true}}, edges::EdgeInfo{true})
    weights = compute_interpolation_weights_vertex_to_edge(vertices, edges)
    return VertexToEdgeInterpolation(VertexToEdgeWeighted(vertices.n, edges.indices.vertices, weights))
end

VertexToEdgeInterpolation(mesh::VoronoiMesh{true}) = VertexToEdgeInterpolation(mesh.vertices.base, mesh.edges)

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

function VertexToEdgePiecewise(vertices::Union{<:VertexBase{true}, <:VertexInfo{true}}, edges::EdgeInfo{true})
    weights = compute_interpolation_weights_vertex_to_edge(vertices, edges)
    weights .= reverse.(weights)
    return VertexToEdgePiecewise(VertexToEdgeWeighted(vertices.n, edges.indices.vertices, weights))
end

VertexToEdgePiecewise(mesh::VoronoiMesh{true}) = VertexToEdgePiecewise(mesh.vertices.base, mesh.edges)

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
    @parallel for e in eachindex(voe)
        @inbounds begin
        v1, v2 = voe[e]
        a1 = areaTriangles[v1]
        a2 = areaTriangles[v2]
        at = a1 + a2
        w1 = a1 / at
        w2 = 1 - w1
        weights[e] = (w1, w2)
        end # inbounds
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

out_eltype(::CellToEdgeMean, in_file::AbstractArray{T}, op::F = Base.identity) where {T, F <: Function} = Base.promote_op(/, Base.promote_op(op, T), Int)

transformation_function!(out_field::AbstractArray, in_field::AbstractArray, Vop::CellToEdgeMean, op::F = Base.identity) where {F <: Function} = to_mean_transformation!(out_field, in_field, Vop.indices, op)

transformation_function!(out_field::AbstractArray, opt_out::F, in_field::AbstractArray, Vop::CellToEdgeMean, op::F2 = Base.identity) where {F <: Function, F2 <: Function} = to_mean_transformation!(out_field, opt_out, in_field, Vop.indices, op)

CellToEdgeMean(cells::Union{<:CellBase, <:CellInfo}, edges::Union{<:EdgeBase, <:EdgeInfo}) = CellToEdgeMean(cells.n, edges.indices.cells)
CellToEdgeMean(mesh::VoronoiMesh) = CellToEdgeMean(mesh.cells.base, mesh.edges.base)

struct CellToEdgeBaricentric{TI, TF} <: CellToEdgeTransformation
    n::Int
    indices::Vector{NTuple{3, TI}}
    weights::Vector{NTuple{3, TF}}
end

function compute_baricentric_cell_to_edge_periodic!(w, inds, edge_pos, cell_pos, v_pos, areaTriangle, verticesOnEdge, cellsOnVertex, x_period::Number, y_period::Number)

    @parallel for e in eachindex(verticesOnEdge)
        @inbounds begin
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
        inds[e] = (i1, i2, i3)
        end # inbounds
    end

    return w, inds
end

function compute_baricentric_cell_to_edge_periodic(edge_pos, cell_pos, vertex_pos, areaTriangle, verticesOnEdge, cellsOnVertex, x_period::Number, y_period::Number)
    TF = eltype(cell_pos.x)
    w = Vector{NTuple{3, TF}}(undef, length(verticesOnEdge))
    inds = Vector{NTuple{3, eltype(eltype(verticesOnEdge))}}(undef, length(verticesOnEdge))
    return compute_baricentric_cell_to_edge_periodic!(w, inds, edge_pos, cell_pos, vertex_pos, areaTriangle, verticesOnEdge, cellsOnVertex, x_period, y_period)
end

function compute_baricentric_cell_to_edge(m::VoronoiMesh{false})
    compute_baricentric_cell_to_edge_periodic(
        m.edges.position, m.cells.position, m.vertices.position, m.vertices.area,
        m.edges.indices.vertices, m.vertices.indices.cells,
        m.attributes[:x_period]::Float64, m.attributes[:y_period]::Float64
    )
end

function compute_baricentric_cell_to_edge!(w, inds, cell_pos, v_pos, areaTriangle, verticesOnEdge, cellsOnVertex)
    R = norm(edge_pos[1])

    @parallel for e in eachindex(verticesOnEdge)
        @inbounds begin
        v1, v2 = verticesOnEdge[e]
        v1_pos = v_pos[v1]
        v2_pos = v_pos[v2]

        e_mid = arc_midpoint(R, v1_pos, v2_pos)

        c11, c12, c13 = cellsOnVertex[v1]
        c11_pos = cell_pos[c11]
        c12_pos = cell_pos[c12]
        c13_pos = cell_pos[c13]

        in_here = in_spherical_triangle(R, e_mid, c11_pos, c12_pos, c13_pos)

        if in_here
            v = v1
            i1 = c11
            i2 = c12
            i3 = c13

            c1_pos = c11_pos
            c2_pos = c12_pos
            c3_pos = c13_pos
        else
            v = v2
            c21, c22, c23 = cellsOnVertex[v2]
            i1 = c21
            i2 = c22
            i3 = c23

            c1_pos = cell_pos[c21]
            c2_pos = cell_pos[c22]
            c3_pos = cell_pos[c23]
        end
        inv_at = inv(areaTriangle[v])

        w1 = spherical_polygon_area(R,e_mid, c1_pos, c2_pos)
        w2 = spherical_polygon_area(R,e_mid, c2_pos, c3_pos)
        w3 = spherical_polygon_area(R,e_mid, c3_pos, c1_pos)

        w1 *= inv_at
        #w2 *= inv_at
        w3 *= inv_at

        #make sure it most likely sums to one (might not due to float point precision)
        w2 = 1 - (w1 + w3)

        w[e] = (w1, w2, w3)
        inds[e] = (i1, i2, i3)
        end # inbounds
    end

    return w, inds
end

function compute_baricentric_cell_to_edge(cell_pos, vertex_pos, areaTriangle, verticesOnEdge, cellsOnVertex)
    TF = eltype(cell_pos.x)
    w = Vector{NTuple{3, TF}}(undef, length(verticesOnEdge))
    inds = Vector{NTuple{3, eltype(eltype(verticesOnEdge))}}(undef, length(verticesOnEdge))
    return compute_baricentric_cell_to_edge!(w, inds, cell_pos, vertex_pos, areaTriangle, verticesOnEdge, cellsOnVertex)
end

function compute_baricentric_cell_to_edge(m::VoronoiMesh{true})
    compute_baricentric_cell_to_edge(
        m.cells.position, m.vertices.position, m.vertices.area,
        m.edges.indices.vertices, m.vertices.indices.cells
    )
end

function CellToEdgeBaricentric(m::VoronoiMesh)
    weights, cellsOnEdgeBaricentric = compute_baricentric_cell_to_edge(m)
    ncells = m.nCells
    return CellToEdgeBaricentric(ncells, cellsOnEdgeBaricentric, weights)
end
