abstract type VertexToEdgeTransformation end

abstract type CellToEdgeTransformation end

struct VertexToEdgeMean{TI} <: VertexToEdgeTransformation
    n::Int
    verticesOnEdge::Vector{NTuple{2,TI}}
end

VertexToEdgeMean(vertices::Union{<:VertexBase,<:VertexInfo},edges::Union{<:EdgeBase,<:EdgeInfo}) = VertexToEdgeMean(vertices.n,edges.indices.vertices)
VertexToEdgeMean(mesh::VoronoiMesh) = VertexToEdgeMean(mesh.vertices.base,mesh.edges.base)

@inbounds @inline (v2e::VertexToEdgeMean)(v_field::AbstractArray{<:Any,N},inds::Vararg{T,N}) where {N,T<:Integer} = to_mean_transformation(v_field,inds,v2e.verticesOnEdge)

function (v2e::VertexToEdgeMean)(e_field::AbstractArray,v_field::AbstractArray)
    is_proper_size(v_field,v2e.n) || throw(DomainError(v_field,"Input array doesn't seem to be a vertex field"))
    is_proper_size(e_field,length(v2e.verticesOnEdge)) || throw(DomainError(e_field,"Output array doesn't seem to be an edge field"))

    to_mean_transformation!(e_field,v_field,v2e.verticesOnEdge)
    
    return e_field
end

function (v2e::VertexToEdgeMean)(v_field::AbstractArray)
    is_proper_size(v_field,v2e.n) || throw(DomainError(v_field,"Input array doesn't seem to be a vertex field"))
    s = construct_new_node_index(size(v_field)...,length(v2e.verticesOnEdge))
    e_field = similar(v_field,s)
    return v2e(e_field,v_field)
end

function (v2e::VertexToEdgeMean)(e_field::AbstractArray,op::F,v_field::AbstractArray) where {F<:Function}
    is_proper_size(v_field,v2e.n) || throw(DomainError(v_field,"Input array doesn't seem to be a vertex field"))
    is_proper_size(e_field,length(v2e.verticesOnEdge)) || throw(DomainError(e_field,"Output array doesn't seem to be an edge field"))

    to_mean_transformation!(e_field,op,v_field,v2e.verticesOnEdge)
   
    return e_field
end

struct CellToEdgeMean{TI} <: CellToEdgeTransformation
    n::Int
    cellsOnEdge::Vector{NTuple{2,TI}}
end

CellToEdgeMean(cells::Union{<:CellBase,<:CellInfo},edges::Union{<:EdgeBase,<:EdgeInfo}) = CellToEdgeMean(cells.n,edges.indices.cells)
CellToEdgeMean(mesh::VoronoiMesh) = CellToEdgeMean(mesh.cells.base,mesh.edges.base)

@inbounds @inline (c2e::CellToEdgeMean)(c_field::AbstractArray{<:Any,N},inds::Vararg{T,N}) where {N,T<:Integer} = to_mean_transformation(c_field,inds,c2e.cellsOnEdge)

function (c2e::CellToEdgeMean)(e_field::AbstractArray,c_field::AbstractArray)
    is_proper_size(c_field,c2e.n) || throw(DomainError(c_field,"Input array doesn't seem to be a cell field"))
    is_proper_size(e_field,length(c2e.cellsOnEdge)) || throw(DomainError(e_field,"Output array doesn't seem to be an edge field"))

    to_mean_transformation!(e_field,c_field,c2e.cellsOnEdge)
    
    return e_field
end

function (c2e::CellToEdgeMean)(c_field::AbstractArray)
    is_proper_size(c_field,c2e.n) || throw(DomainError(c_field,"Input array doesn't seem to be a cell field"))
    s = construct_new_node_index(size(c_field)...,length(c2e.cellsOnEdge))
    e_field = similar(c_field,s)
    return c2e(e_field,c_field)
end

function (c2e::CellToEdgeMean)(e_field::AbstractArray,op::F,c_field::AbstractArray) where {F<:Function}
    is_proper_size(c_field,c2e.n) || throw(DomainError(c_field,"Input array doesn't seem to be a cell field"))
    is_proper_size(e_field,length(c2e.cellsOnEdge)) || throw(DomainError(e_field,"Output array doesn't seem to be an edge field"))

    to_mean_transformation!(e_field,op,c_field,c2e.cellsOnEdge)

    return e_field
end

struct CellToEdgeBaricentric{TI,TF} <: CellToEdgeTransformation
    n::Int
    cellsOnEdgeBaricentric::Vector{NTuple{3,TI}}
    weights::Vector{NTuple{3,TF}}
end

function compute_baricentric_cell_to_edge_periodic!(w,inds,edge_pos,cell_pos,areaTriangle,verticesOnEdge,cellsOnVertex,x_period::Number,y_period::Number)
    
    @inbounds for e in eachindex(verticesOnEdge)
        e_pos = edge_pos[e]
        v1,v2 = verticesOnEdge[e]

        c11,c12,c13 = cellsOnVertex[v1]
        c11_pos = closest(e_pos,cell_pos[c11],x_period,y_period)
        c12_pos = closest(e_pos,cell_pos[c12],x_period,y_period)
        c13_pos = closest(e_pos,cell_pos[c13],x_period,y_period)

        at = areaTriangle[v1]

        in_here, w1,w2,w3,_ = in_triangle(e_pos,at,c11_pos,c12_pos,c13_pos)
        if in_here
            i1 = c11
            i2 = c12
            i3 = c13
        else
            c21,c22,c23 = cellsOnVertex[v2]
            c21_pos = closest(e_pos,cell_pos[c21],x_period,y_period)
            c22_pos = closest(e_pos,cell_pos[c22],x_period,y_period)
            c23_pos = closest(e_pos,cell_pos[c23],x_period,y_period)

            at = areaTriangle[v2]
            in_here, w1,w2,w3,_ = in_triangle(e_pos,at,c21_pos,c22_pos,c23_pos)
            i1 = c21
            i2 = c22
            i3 = c23
        end
        inv_at = inv(at)
        w1 *= inv_at
        w2 *= inv_at
        w3 *= inv_at

        #make sure it most likely sums to one (might not due to float point precision)
        w2 = 1 - (w1 + w3)

        w[e] = (w1,w2,w3)
        inds[e] = map(Int,(i1,i2,i3))
    end

    return w,inds
end

function compute_baricentric_cell_to_edge_periodic(edge_pos,cell_pos,areaTriangle,verticesOnEdge,cellsOnVertex,x_period::Number,y_period::Number)
    TF = eltype(cell_pos.x)
    w = Vector{NTuple{3,TF}}(undef,length(verticesOnEdge))
    inds = Vector{NTuple{3,Int}}(undef,length(verticesOnEdge))
    return compute_baricentric_cell_to_edge_periodic!(w,inds,edge_pos,cell_pos,areaTriangle,verticesOnEdge,cellsOnVertex,x_period,y_period)
end

function compute_baricentric_cell_to_edge_periodic(m::VoronoiMesh)
    compute_baricentric_cell_to_edge_periodic(m.edges.position, m.cells.position,m.vertices.area,
                                              m.edges.indices.vertices, m.vertices.indices.cells,
                                              m.attributes[:x_period]::Float64, m.attributes[:y_period]::Float64)
end

function CellToEdgeBaricentric(m::VoronoiMesh{false})
    weights, cellsOnEdgeBaricentric = compute_baricentric_cell_to_edge_periodic(m) 
    ncells = m.nCells
    return CellToEdgeBaricentric(ncells,cellsOnEdgeBaricentric,weights)
end

function (c2e::CellToEdgeBaricentric)(e_field::AbstractArray,c_field::AbstractArray)
    is_proper_size(c_field,c2e.n) || throw(DomainError(c_field,"Input array doesn't seem to be a cell field"))
    is_proper_size(e_field,length(c2e.cellsOnEdgeBaricentric)) || throw(DomainError(e_field,"Output array doesn't seem to be an edge field"))

    weighted_sum_transformation!(e_field,c_field,c2e.weights, c2e.cellsOnEdgeBaricentric)
    
    return e_field
end

function (c2e::CellToEdgeBaricentric)(c_field::AbstractArray)
    is_proper_size(c_field,c2e.n) || throw(DomainError(c_field,"Input array doesn't seem to be a cell field"))
    s = construct_new_node_index(size(c_field)...,length(c2e.cellsOnEdgeBaricentric))
    e_field = similar(c_field,s)
    return c2e(e_field,c_field)
end

function (c2e::CellToEdgeBaricentric)(e_field::AbstractArray,op::F,c_field::AbstractArray) where {F<:Function}
    is_proper_size(c_field,c2e.n) || throw(DomainError(c_field,"Input array doesn't seem to be a cell field"))
    is_proper_size(e_field,length(c2e.cellsOnEdgeBaricentric)) || throw(DomainError(e_field,"Output array doesn't seem to be an edge field"))

    weighted_sum_transformation!(e_field, op, c_field, c2e.weights, c2e.cellsOnEdgeBaricentric)

    return e_field
end
