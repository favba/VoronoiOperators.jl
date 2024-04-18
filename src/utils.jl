@inline get_node_index(i::Integer) = i
@inline get_node_index(k::Integer,i::Integer) = i
@inline get_node_index(k::Integer,i::Integer,t::Integer) = i

@inline construct_new_node_index(i::Integer,n::Integer) = (oftype(i,n),)
@inline construct_new_node_index(k::Integer,i::Integer,n::Integer) = (k,oftype(i,n))
@inline construct_new_node_index(k::Integer,i::Integer,t::Integer,n::Integer) = (k,oftype(i,n),t)

@inline is_proper_size(field::AbstractVector,n::Integer) = length(field) == n
@inline is_proper_size(field::AbstractMatrix,n::Integer) = size(field,2) == n
@inline is_proper_size(field::AbstractArray{<:Any,3},n::Integer) = size(field,2) == n

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
