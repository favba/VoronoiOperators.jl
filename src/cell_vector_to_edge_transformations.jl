abstract type VecCellToEdgeTransformation end

@inbounds @inline function vec_to_edge_mean_transformation(n::Vec, input_field::AbstractArray{<:Any,N}, inds1::NTuple{N,T1}, inds2::NTuple{N,T2}) where {N,T1<:Integer,T2<:Integer}
    @inline begin
       mean_val = dot(n, 0.5*(input_field[inds1...] + input_field[inds2...]))
    end
    return mean_val
end

@inbounds @inline function vec_to_edge_mean_transformation(n::VecArray, input_field::AbstractArray{<:Any,N}, inds::NTuple{N,T1}, indices::AbstractVector{NTuple{2,T2}}) where {N,T1<:Integer,T2<:Integer}
    @inline begin
        e_i = get_node_index(inds...)
        v1, v2 = indices[e_i]
        ind1 = construct_new_node_index(inds...,v1)
        ind2 = construct_new_node_index(inds...,v2)
    end
    return vec_to_edge_mean_transformation(n[e_i],input_field,ind1,ind2)
end

function vec_to_edge_mean_transformation!(output_field::AbstractVector,input_field::AbstractVector{<:Vec},n::VecArray,indices::AbstractVector{NTuple{2,T2}}) where {T2<:Integer}
    @inbounds for i in eachindex(output_field)
        output_field[i] = vec_to_edge_mean_transformation(n,input_field,(i,),indices)
    end
    return output_field
end

function vec_to_edge_mean_transformation!(output_field::AbstractVector,op::F,input_field::AbstractVector{<:Vec},n::VecArray,indices::AbstractVector{NTuple{2,T2}}) where {F<:Function,T2<:Integer}
    @inbounds for i in eachindex(output_field)
        output_field[i] = op(output_field[i], vec_to_edge_mean_transformation(n,input_field,(i,),indices))
    end
    return output_field
end

function vec_to_edge_mean_transformation!(output_field::AbstractMatrix,input_field::AbstractMatrix{<:Vec},n::VecArray,indices::AbstractVector{NTuple{2,T2}}) where {T2<:Integer}

    @inbounds for i in axes(output_field,2)
        v1, v2 = map(Int,indices[i])
        n_i = n[i]
        for k in axes(output_field,1)
            ind1 = (k,v1)
            ind2 = (k,v2)
            output_field[k,i] = vec_to_edge_mean_transformation(n_i,input_field,ind1,ind2)
        end
    end

    return output_field
end

function vec_to_edge_mean_transformation!(output_field::AbstractMatrix,op::F,input_field::AbstractMatrix{<:Vec},n::VecArray,indices::AbstractVector{NTuple{2,T2}}) where {F<:Function,T2<:Integer}

    @inbounds for i in axes(output_field,2)
        v1, v2 = map(Int,indices[i])
        n_i = n[i]
        for k in axes(output_field,1)
            ind1 = (k,v1)
            ind2 = (k,v2)
            output_field[k,i] = op(output_field[k,i], vec_to_edge_mean_transformation(n_i, input_field,ind1,ind2))
        end
    end

    return output_field
end

function vec_to_edge_mean_transformation!(output_field::AbstractArray{<:Any,3},input_field::AbstractArray{<:Vec,3},n::VecArray,indices::AbstractVector{NTuple{2,T2}}) where {T2<:Integer}

    @inbounds for t in axes(output_field,3)
        for i in axes(output_field,2)
            v1, v2 = map(Int,indices[i])
            n_i = n[i]
            for k in axes(output_field,1)
                ind1 = (k,v1,t)
                ind2 = (k,v2,t)
                output_field[k,i,t] = vec_to_edge_mean_transformation(n_i,input_field,ind1,ind2)
            end
        end
    end

    return output_field
end

function vec_to_edge_mean_transformation!(output_field::AbstractArray{<:Any,3},op::F,input_field::AbstractArray{<:Vec,3},n::VecArray,indices::AbstractVector{NTuple{2,T2}}) where {F<:Function,T2<:Integer}

    @inbounds for t in axes(output_field,3)
        for i in axes(output_field,2)
            v1, v2 = map(Int,indices[i])
            n_i = n[i]
            for k in axes(output_field,1)
                ind1 = (k,v1,t)
                ind2 = (k,v2,t)
                output_field[k,i,t] = op(output_field[k,i,t], vec_to_edge_mean_transformation(n_i, input_field,ind1,ind2))
            end
        end
    end

    return output_field
end

struct VecCellToEdgeMean{TI,TF,Tz} <: VecCellToEdgeTransformation
    n::Int
    cellsOnEdge::Vector{NTuple{2,TI}}
    normalVectors::TensorsLite.VecMaybe2DxyArray{TF,Tz,1}
end

VecCellToEdgeMean(cells::Union{<:CellBase,<:CellInfo},edges::EdgeInfo) = VecCellToEdgeMean(cells.n,edges.indices.cells,edges.normalVectors)

function VecCellToEdgeMean(mesh::VoronoiMesh) 
    isdefined(mesh.edges,:normalVectors) || compute_edge_normals!(mesh)
    VecCellToEdgeMean(mesh.cells.base,mesh.edges)
end

@inbounds @inline (vc2e::VecCellToEdgeMean)(c_field::AbstractArray{<:Any,N},inds::Vararg{T,N}) where {N,T<:Integer} = vec_to_edge_mean_transformation(vc2e.normalVectors,c_field,inds,vc2e.cellsOnEdge)

function (vc2e::VecCellToEdgeMean)(e_field::AbstractArray,c_field::AbstractArray{<:Vec})
    is_proper_size(c_field,vc2e.n) || throw(DomainError(c_field,"Input array doesn't seem to be a cell field"))
    is_proper_size(e_field,length(vc2e.cellsOnEdge)) || throw(DomainError(e_field,"Output array doesn't seem to be an edge field"))

    vec_to_edge_mean_transformation!(e_field,c_field,vc2e.normalVectors,vc2e.cellsOnEdge)
    
    return e_field
end

function (vc2e::VecCellToEdgeMean)(c_field::AbstractArray{<:Vec})
    is_proper_size(c_field,vc2e.n) || throw(DomainError(c_field,"Input array doesn't seem to be a cell field"))
    s = construct_new_node_index(size(c_field)...,length(vc2e.cellsOnEdge))
    e_field = Array{nonzero_eltype(eltype(c_field))}(undef,s)
    return vc2e(e_field,c_field)
end

function (vc2e::VecCellToEdgeMean)(e_field::AbstractArray,op::F,c_field::AbstractArray) where {F<:Function}
    is_proper_size(c_field,vc2e.n) || throw(DomainError(c_field,"Input array doesn't seem to be a cell field"))
    is_proper_size(e_field,length(vc2e.cellsOnEdge)) || throw(DomainError(e_field,"Output array doesn't seem to be an edge field"))

    vec_to_edge_mean_transformation!(e_field,op,c_field,vc2e.normalVectors,vc2e.cellsOnEdge)

    return e_field
end
