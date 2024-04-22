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

@inline function dropfirst(t::NTuple{N}) where N
    return ntuple(i->t[i+1],Val{N-1}())
end

@inbounds @inline function weighted_sum(input_field::AbstractArray,weights::NTuple{2},indices::NTuple{2})
   muladd(weights[1],input_field[indices[1]...],weights[2]*input_field[indices[2]...]) 
end

#@inbounds @inline function weighted_sum(weights::NTuple{2},vals::NTuple{2})
#   muladd(weights[1],vals[1],weights[2]*vals[2]) 
#end

for N = 3:24
    @eval function weighted_sum(input_field::AbstractArray,weights::NTuple{$N},indices::NTuple{$N})
        $(Expr(:meta,:inline))
        @inbounds muladd(weights[1], input_field[indices[1]...], weighted_sum(input_field, dropfirst(weights), dropfirst(indices)))
    end
#    @eval function weighted_sum(weights::NTuple{$N},vals::NTuple{$N})
#        $(Expr(:meta,:inline))
#        @inbounds muladd(weights[1], vals[1], weighted_sum(dropfirst(weights), dropfirst(vals)))
#    end
end

function weighted_sum(input_field::AbstractArray,weights,indices)
    r = zero(eltype(input_field))
    @inbounds for (i,inds) in enumerate(indices)
        r = muladd(weights[i],input_field[inds...],r)
    end
    return r
end

@inline insert_index(inds_input::NTuple{0},inds_output::NTuple{N,Int}) where N = inds_output
@inline insert_index(inds_input::NTuple{1},inds_output::NTuple{N,Int}) where N = ntuple(i->(inds_input[1],inds_output[i]),Val{N}())
@inline insert_index(inds_input::NTuple{2},inds_output::NTuple{N,Int}) where N = ntuple(i->(inds_input[1],inds_output[i],inds_input[2]),Val{N}())

function weighted_sum_transformation!(output_field::AbstractVector,input_field::AbstractVector,weights,indices::AbstractVector{NTuple{N,T2}}) where {N,T2<:Integer}
    @inbounds for i in eachindex(output_field)
        output_field[i] = weighted_sum(input_field,weights[i],indices[i])
    end
    return output_field
end

function weighted_sum_transformation!(output_field::AbstractVector,op::F,input_field::AbstractVector,weights,indices::AbstractVector{NTuple{N,T2}}) where {N,F<:Function,T2<:Integer}
    @inbounds for i in eachindex(output_field)
        output_field[i] = op(output_field[i], weighted_sum(input_field,weights[i],indices[i]))
    end
    return output_field
end

function weighted_sum_transformation!(output_field::AbstractMatrix,input_field::AbstractMatrix,weights,indices::AbstractVector{NTuple{N,T2}}) where {N,T2<:Integer}

    @inbounds for i in axes(output_field,2)
        inds = indices[i]
        w = weights[i]
        for k in axes(output_field,1)
            inds_input = insert_index((k,),inds)
            output_field[k,i] = weighted_sum(input_field,w,inds_input)
        end
    end

    return output_field
end

function weighted_sum_transformation!(output_field::AbstractMatrix,op::F,input_field::AbstractMatrix,weights,indices::AbstractVector{NTuple{N,T2}}) where {F<:Function,N,T2<:Integer}

    @inbounds for i in axes(output_field,2)
        inds = indices[i]
        w = weights[i]
        for k in axes(output_field,1)
            inds_input = insert_index((k,),inds)
            output_field[k,i] = op(output_field[k,i],weighted_sum(input_field,w,inds_input))
        end
    end

    return output_field
end

function weighted_sum_transformation!(output_field::AbstractArray{<:Any,3},input_field::AbstractArray{<:Any,3},weights,indices::AbstractVector{NTuple{N,T2}}) where {N,T2<:Integer}

    @inbounds for i in axes(output_field,2)
        inds = indices[i]
        w = weights[i]
        for t in axes(output_field,3)
            for k in axes(output_field,1)
                inds_input = insert_index((k,t),inds)
                output_field[k,i,t] = weighted_sum(input_field,w,inds_input)
            end
        end
    end

    return output_field
end

function weighted_sum_transformation!(output_field::AbstractArray{<:Any,3},op::F,input_field::AbstractArray{<:Any,3},weights,indices::AbstractVector{NTuple{N,T2}}) where {F<:Function,N,T2<:Integer}

    @inbounds for i in axes(output_field,2)
        inds = indices[i]
        w = weights[i]
        for t in axes(output_field,3)
            for k in axes(output_field,1)
                inds_input = insert_index((k,t),inds)
                output_field[k,i,t] = op(output_field[k,i,t],weighted_sum(input_field,w,inds_input))
            end
        end
    end

    return output_field
end
