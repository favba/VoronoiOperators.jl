abstract type VertexToEdgeTransformation <: LinearVoronoiOperator end

name_input(::VertexToEdgeTransformation) = "vertex"
name_output(::VertexToEdgeTransformation) = "edge"

struct VertexToEdgeMean{TI} <: VertexToEdgeTransformation
    n::Int
    indices::Vector{NTuple{2, TI}}
end

out_eltype(::VertexToEdgeMean, in_file::AbstractArray{T}, op::F = Base.identity) where {T, F} = Base.promote_op(/, Base.promote_op(op, T), Int)

transformation_function!(out_field::AbstractArray, in_field::AbstractArray, Vop::VertexToEdgeMean, op::F) where {F <: Function} = to_mean_transformation!(out_field, in_field, Vop.indices, op)

transformation_function!(out_field::AbstractArray, opt_out::F, in_field::AbstractArray, Vop::VertexToEdgeMean, op::F2) where {F <: Function, F2 <: Function} = to_mean_transformation!(out_field, opt_out, in_field, Vop.indices, op)

VertexToEdgeMean(vertices::Vertices, edges::Edges) = VertexToEdgeMean(vertices.n, edges.vertices)
VertexToEdgeMean(mesh::AbstractVoronoiMesh) = VertexToEdgeMean(mesh.vertices, mesh.edges)

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

function compute_interpolation_weights_vertex_to_edge(R::Real, epos, vpos, voe, dvEdge)
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

function compute_interpolation_weights_vertex_to_edge(vertices::Vertices{false}, edges::Edges{false}, xp::Number, yp::Number)
    return compute_interpolation_weights_vertex_to_edge_periodic(edges.position, vertices.position, edges.vertices, edges.length, xp, yp)
end

function VertexToEdgeInterpolation(vertices::Vertices{false}, edges::Edges{false}, xp::Number, yp::Number)
    weights = compute_interpolation_weights_vertex_to_edge(vertices, edges, xp, yp)
    return VertexToEdgeInterpolation(VertexToEdgeWeighted(vertices.n, edges.vertices, weights))
end

VertexToEdgeInterpolation(mesh::AbstractVoronoiMesh{false}) = VertexToEdgeInterpolation(mesh.vertices, mesh.edges, mesh.x_period, mesh.y_period)

function compute_interpolation_weights_vertex_to_edge(vertices::Vertices{true}, edges::Edges{true})
    return compute_interpolation_weights_vertex_to_edge(edges.sphere_radius, edges.position, vertices.position, edges.vertices, edges.length)
end

function VertexToEdgeInterpolation(vertices::Vertices{true}, edges::Edges{true})
    weights = compute_interpolation_weights_vertex_to_edge(vertices, edges)
    return VertexToEdgeInterpolation(VertexToEdgeWeighted(vertices.n, edges.vertices, weights))
end

VertexToEdgeInterpolation(mesh::AbstractVoronoiMesh{true}) = VertexToEdgeInterpolation(mesh.vertices, mesh.edges)

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

function VertexToEdgePiecewise(vertices::Vertices{false}, edges::Edges{false}, xp::Number, yp::Number)
    weights = compute_interpolation_weights_vertex_to_edge(vertices, edges, xp, yp)
    weights .= reverse.(weights)
    return VertexToEdgePiecewise(VertexToEdgeWeighted(vertices.n, edges.vertices, weights))
end

VertexToEdgePiecewise(mesh::AbstractVoronoiMesh{false}) = VertexToEdgePiecewise(mesh.vertices, mesh.edges, mesh.x_period, mesh.y_period)

function VertexToEdgePiecewise(vertices::Vertices{true}, edges::Edges{true})
    weights = compute_interpolation_weights_vertex_to_edge(vertices, edges)
    weights .= reverse.(weights)
    return VertexToEdgePiecewise(VertexToEdgeWeighted(vertices.n, edges.vertices, weights))
end

VertexToEdgePiecewise(mesh::AbstractVoronoiMesh{true}) = VertexToEdgePiecewise(mesh.vertices, mesh.edges)

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

function VertexToEdgeArea(vertices::Vertices, edges::Edges)
    weights = compute_area_weights_vertex_to_edge(edges.vertices, vertices.area)
    return VertexToEdgeArea(VertexToEdgeWeighted(vertices.n, edges.vertices, weights))
end

VertexToEdgeArea(mesh::AbstractVoronoiMesh) = VertexToEdgeArea(mesh.vertices, mesh.edges)

abstract type CellToEdgeTransformation <: LinearVoronoiOperator end

name_input(::CellToEdgeTransformation) = "cell"
name_output(::CellToEdgeTransformation) = "edge"

struct CellToEdgeMean{TI} <: CellToEdgeTransformation
    n::Int
    indices::Vector{NTuple{2, TI}}
end

out_eltype(::CellToEdgeMean, in_file::AbstractArray{T}, op::F = Base.identity) where {T, F} = Base.promote_op(/, Base.promote_op(op, T), Int)

transformation_function!(out_field::AbstractArray, in_field::AbstractArray, Vop::CellToEdgeMean, op::F = Base.identity) where {F <: Function} = to_mean_transformation!(out_field, in_field, Vop.indices, op)

transformation_function!(out_field::AbstractArray, opt_out::F, in_field::AbstractArray, Vop::CellToEdgeMean, op::F2 = Base.identity) where {F <: Function, F2 <: Function} = to_mean_transformation!(out_field, opt_out, in_field, Vop.indices, op)

CellToEdgeMean(cells::Cells, edges::Edges) = CellToEdgeMean(cells.n, edges.cells)
CellToEdgeMean(mesh::AbstractVoronoiMesh) = CellToEdgeMean(mesh.cells, mesh.edges)

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

function compute_baricentric_cell_to_edge(m::AbstractVoronoiMesh{false})
    compute_baricentric_cell_to_edge_periodic(
        m.edges.position, m.cells.position, m.vertices.position, m.vertices.area,
        m.edges.vertices, m.vertices.cells,
        m.x_period, m.y_period
    )
end

function compute_baricentric_cell_to_edge!(R, w, inds, cell_pos, v_pos, areaTriangle, verticesOnEdge, cellsOnVertex)

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

function compute_baricentric_cell_to_edge(R, cell_pos, vertex_pos, areaTriangle, verticesOnEdge, cellsOnVertex)
    TF = eltype(cell_pos.x)
    w = Vector{NTuple{3, TF}}(undef, length(verticesOnEdge))
    inds = Vector{NTuple{3, eltype(eltype(verticesOnEdge))}}(undef, length(verticesOnEdge))
    return compute_baricentric_cell_to_edge!(R, w, inds, cell_pos, vertex_pos, areaTriangle, verticesOnEdge, cellsOnVertex)
end

function compute_baricentric_cell_to_edge(m::AbstractVoronoiMesh{true})
    compute_baricentric_cell_to_edge(
        m.sphere_radius, m.cells.position, m.vertices.position, m.vertices.area,
        m.edges.vertices, m.vertices.cells
    )
end

function CellToEdgeBaricentric(m::AbstractVoronoiMesh)
    weights, cellsOnEdgeBaricentric = compute_baricentric_cell_to_edge(m)
    ncells = m.cells.n
    return CellToEdgeBaricentric(ncells, cellsOnEdgeBaricentric, weights)
end
