#Should always have verticesOnEdge
abstract type VertexToEdgeTransformation end

#Should always have cellsOnEdge
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

