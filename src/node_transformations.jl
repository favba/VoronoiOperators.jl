#Should always have verticesOnEdge
abstract type VertexToEdgeTransformation end

#Should always have cellsOnEdge
abstract type CellToEdgeTransformation end

@inbounds @inline function to_mean_transformation(input_field::AbstractArray{<:Any,N}, inds1::NTuple{N,T1}, inds2::NTuple{N,T2}) where {N,T1<:Integer,T2<:Integer}
    @inline begin
       mean_val = 0.5*(input_field[inds1...] + input_field[inds2...])
    end
    return mean_val
end

@inbounds @inline function to_mean_transformation(input_field::AbstractArray{<:Any,N}, inds::NTuple{N,T1}, indices::AbstractVector{NTuple{2,T2}}) where {N,T1<:Integer,T2<:Integer}
    @inline begin
        e_i = get_node_index(inds...)
        v1, v2 = indices[e_i]
        ind1 = construct_new_node_index(inds...,v1)
        ind2 = construct_new_node_index(inds...,v2)
    end
    return to_mean_transformation(input_field,ind1,ind2)
end

function to_mean_transformation!(output_field::AbstractVector,input_field::AbstractVector,indices::AbstractVector{NTuple{2,T2}}) where {T2<:Integer}
    @inbounds for i in eachindex(output_field)
        output_field[i] = to_mean_transformation(input_field,(i,),indices)
    end
    return output_field
end

function to_mean_transformation!(output_field::AbstractVector,op::F,input_field::AbstractVector,indices::AbstractVector{NTuple{2,T2}}) where {F<:Function,T2<:Integer}
    @inbounds for i in eachindex(output_field)
        output_field[i] = op(output_field[i], to_mean_transformation(input_field,(i,),indices))
    end
    return output_field
end

function to_mean_transformation!(output_field::AbstractMatrix,input_field::AbstractMatrix,indices::AbstractVector{NTuple{2,T2}}) where {T2<:Integer}

    @inbounds for i in axes(output_field,2)
        v1, v2 = map(Int,indices[i])
        for k in axes(output_field,1)
            ind1 = (k,v1)
            ind2 = (k,v2)
            output_field[k,i] = to_mean_transformation(input_field,ind1,ind2)
        end
    end

    return output_field
end

function to_mean_transformation!(output_field::AbstractMatrix,op::F,input_field::AbstractMatrix,indices::AbstractVector{NTuple{2,T2}}) where {F<:Function,T2<:Integer}

    @inbounds for i in axes(output_field,2)
        v1, v2 = map(Int,indices[i])
        for k in axes(output_field,1)
            ind1 = (k,v1)
            ind2 = (k,v2)
            output_field[k,i] = op(output_field[k,i], to_mean_transformation(input_field,ind1,ind2))
        end
    end

    return output_field
end

function to_mean_transformation!(output_field::AbstractArray{<:Any,3},input_field::AbstractArray{<:Any,3},indices::AbstractVector{NTuple{2,T2}}) where {T2<:Integer}

    @inbounds for t in axes(output_field,3)
        for i in axes(output_field,2)
            v1, v2 = map(Int,indices[i])
            for k in axes(output_field,1)
                ind1 = (k,v1,t)
                ind2 = (k,v2,t)
                output_field[k,i,t] = to_mean_transformation(input_field,ind1,ind2)
            end
        end
    end

    return output_field
end

function to_mean_transformation!(output_field::AbstractArray{<:Any,3},op::F,input_field::AbstractArray{<:Any,3},indices::AbstractVector{NTuple{2,T2}}) where {F<:Function,T2<:Integer}

    @inbounds for t in axes(output_field,3)
        for i in axes(output_field,2)
            v1, v2 = map(Int,indices[i])
            for k in axes(output_field,1)
                ind1 = (k,v1,t)
                ind2 = (k,v2,t)
                output_field[k,i,t] = op(output_field[k,i,t], to_mean_transformation(input_field,ind1,ind2))
            end
        end
    end

    return output_field
end

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
