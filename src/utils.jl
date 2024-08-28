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
    @parallel for i in eachindex(output_field)
        @inbounds output_field[i] = to_mean_transformation(input_field,(i,),indices)
    end
    return output_field
end

function to_mean_transformation!(output_field::AbstractVector,op::F,input_field::AbstractVector,indices::AbstractVector{NTuple{2,T2}}) where {F<:Function,T2<:Integer}
    @parallel for i in eachindex(output_field)
        @inbounds output_field[i] = op(output_field[i], to_mean_transformation(input_field,(i,),indices))
    end
    return output_field
end

function to_mean_transformation!(output_field::AbstractMatrix,input_field::AbstractMatrix,indices::AbstractVector{NTuple{2,T2}}) where {T2<:Integer}

    @parallel for i in axes(output_field,2)
        @inbounds begin
        v1, v2 = map(Int,indices[i])
        for k in axes(output_field,1)
            ind1 = (k,v1)
            ind2 = (k,v2)
            output_field[k,i] = to_mean_transformation(input_field,ind1,ind2)
        end
        end
    end

    return output_field
end

function to_mean_transformation!(output_field::AbstractMatrix,op::F,input_field::AbstractMatrix,indices::AbstractVector{NTuple{2,T2}}) where {F<:Function,T2<:Integer}

    @parallel for i in axes(output_field,2)
        @inbounds begin
        v1, v2 = map(Int,indices[i])
        for k in axes(output_field,1)
            ind1 = (k,v1)
            ind2 = (k,v2)
            output_field[k,i] = op(output_field[k,i], to_mean_transformation(input_field,ind1,ind2))
        end
        end
    end

    return output_field
end

function to_mean_transformation!(output_field::AbstractArray{<:Any,3},input_field::AbstractArray{<:Any,3},indices::AbstractVector{NTuple{2,T2}}) where {T2<:Integer}

    @parallel for t in axes(output_field,3)
        @inbounds for i in axes(output_field,2)
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

    @parallel for t in axes(output_field,3)
        @inbounds for i in axes(output_field,2)
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

@inbounds @inline function weighted_sum(input_field::AbstractVector,weights::NTuple{2},indices::NTuple{2})
   muladd(weights[1],input_field[indices[1]],weights[2]*input_field[indices[2]]) 
end

@inbounds @inline function weighted_sum(input_field::AbstractMatrix,weights::NTuple{2},indices::NTuple{2},k::Int)
   muladd(weights[1],input_field[k,indices[1]],weights[2]*input_field[k,indices[2]]) 
end

@inbounds @inline function weighted_sum(input_field::AbstractArray{<:Any,3},weights::NTuple{2},indices::NTuple{2},k::Int,t::Int)
   muladd(weights[1],input_field[k,indices[1],t],weights[2]*input_field[k,indices[2],t]) 
end

@inbounds @inline function weighted_sum(op::F,input_field::AbstractVector,weights::NTuple{2},indices::NTuple{2}) where F<:Function
   muladd(weights[1],op(input_field[indices[1]]),weights[2]*op(input_field[indices[2]]))
end

@inbounds @inline function weighted_sum(op::F,input_field::AbstractMatrix,weights::NTuple{2},indices::NTuple{2},k::Int) where F<:Function
   muladd(weights[1],op(input_field[k,indices[1]]),weights[2]*op(input_field[k,indices[2]]))
end

@inbounds @inline function weighted_sum(op::F,input_field::AbstractArray{<:Any,3},weights::NTuple{2},indices::NTuple{2},k::Int,t::Int) where F<:Function
   muladd(weights[1],op(input_field[k,indices[1],t]),weights[2]*op(input_field[k,indices[2],t]))
end

@inbounds @inline function weighted_sum(input_field::AbstractVector,weights::NTuple{N},indices::NTuple{N}) where N
    @inbounds muladd(weights[1], input_field[indices[1]], weighted_sum(input_field, dropfirst(weights), dropfirst(indices)))
end

@inbounds @inline function weighted_sum(input_field::AbstractMatrix,weights::NTuple{N},indices::NTuple{N},k::Int) where N
    @inbounds muladd(weights[1], input_field[k,indices[1]], weighted_sum(input_field, dropfirst(weights), dropfirst(indices),k))
end

@inbounds @inline function weighted_sum(input_field::AbstractArray{<:Any,3},weights::NTuple{N},indices::NTuple{N},k::Int,t::Int) where N
    @inbounds muladd(weights[1], input_field[k,indices[1],t], weighted_sum(input_field, dropfirst(weights), dropfirst(indices),k,t))
end

@inbounds @inline function weighted_sum(op::F,input_field::AbstractVector,weights::NTuple{N},indices::NTuple{N}) where {N,F<:Function}
    @inbounds muladd(weights[1], op(input_field[indices[1]]), weighted_sum(op,input_field, dropfirst(weights), dropfirst(indices)))
end

@inbounds @inline function weighted_sum(op::F,input_field::AbstractMatrix,weights::NTuple{N},indices::NTuple{N},k::Int) where {N,F<:Function}
    @inbounds muladd(weights[1], op(input_field[k,indices[1]]), weighted_sum(op,input_field, dropfirst(weights), dropfirst(indices),k))
end

@inbounds @inline function weighted_sum(op::F,input_field::AbstractArray{<:Any,3},weights::NTuple{N},indices::NTuple{N},k::Int,t::Int) where {N,F<:Function}
    @inbounds muladd(weights[1], op(input_field[k,indices[1],t]), weighted_sum(op,input_field, dropfirst(weights), dropfirst(indices),k,t))
end

@inline function weighted_sum(input_field::AbstractVector,weights,indices)
    r = zero(Base.promote_op(*,eltype(weights),eltype(input_field)))
    @inbounds for (i,ind) in enumerate(indices)
        r = muladd(weights[i],input_field[ind],r)
    end
    return r
end

@inline function weighted_sum(input_field::AbstractMatrix,weights,indices,k::Int)
    r = zero(Base.promote_op(*,eltype(weights),eltype(input_field)))
    @inbounds for (i,ind) in enumerate(indices)
        r = muladd(weights[i],input_field[k,ind],r)
    end
    return r
end

@inline function weighted_sum(input_field::AbstractArray{<:Any,3},weights,indices,k::Int,t::Int)
    r = zero(Base.promote_op(*,eltype(weights),eltype(input_field)))
    @inbounds for (i,ind) in enumerate(indices)
        r = muladd(weights[i],input_field[k,ind,t],r)
    end
    return r
end

@inline function weighted_sum(op::F,input_field::AbstractVector,weights,indices) where F<:Function
    r = zero(Base.promote_op(*,eltype(weights),Base.promote_op(op,eltype(input_field))))
    @inbounds for (i,ind) in enumerate(indices)
        r = muladd(weights[i],op(input_field[ind]),r)
    end
    return r
end

@inline function weighted_sum(op::F,input_field::AbstractMatrix,weights,indices,k::Int) where F<:Function
    r = zero(Base.promote_op(*,eltype(weights),Base.promote_op(op,eltype(input_field))))
    @inbounds for (i,ind) in enumerate(indices)
        r = muladd(weights[i],op(input_field[k,ind]),r)
    end
    return r
end

@inline function weighted_sum(op::F,input_field::AbstractArray{<:Any,3},weights,indices,k::Int,t::Int) where F<:Function
    r = zero(Base.promote_op(*,eltype(weights),Base.promote_op(op,eltype(input_field))))
    @inbounds for (i,ind) in enumerate(indices)
        r = muladd(weights[i],op(input_field[k,ind,t]),r)
    end
    return r
end

@inline insert_index(inds_input::NTuple{0},inds_output::NTuple{N,T}) where {N,T<:Integer} = map(Int,inds_output)
@inline insert_index(inds_input::NTuple{1},inds_output::NTuple{N,T}) where {N,T<:Integer} = ntuple(i->(Int(inds_input[1]),Int(inds_output[i])),Val{N}())
@inline insert_index(inds_input::NTuple{2},inds_output::NTuple{N,T}) where {N,T<:Integer} = ntuple(i->(Int(inds_input[1]),Int(inds_output[i]),Int(inds_input[2])),Val{N}())

function weighted_sum_transformation!(output_field::AbstractVector,input_field::AbstractVector,weights,indices)
    @parallel for i in eachindex(output_field)
        @inbounds output_field[i] = weighted_sum(input_field,weights[i],indices[i])
    end
    return output_field
end

function weighted_sum_transformation!(output_field::AbstractVector,input_field::AbstractVector,op::F,weights,indices) where F<:Function
    @parallel for i in eachindex(output_field)
        @inbounds output_field[i] = weighted_sum(op,input_field,weights[i],indices[i])
    end
    return output_field
end

function weighted_sum_transformation!(output_field::AbstractVector,op::F,input_field::AbstractVector,weights,indices) where F<:Function
    @parallel for i in eachindex(output_field)
        @inbounds output_field[i] = op(output_field[i], weighted_sum(input_field,weights[i],indices[i]))
    end
    return output_field
end

function weighted_sum_transformation!(output_field::AbstractVector,op::F,input_field::AbstractVector,op2::F2,weights,indices) where {F<:Function,F2<:Function}
    @parallel for i in eachindex(output_field)
        @inbounds output_field[i] = op(output_field[i], weighted_sum(op2,input_field,weights[i],indices[i]))
    end
    return output_field
end

function weighted_sum_transformation!(output_field::AbstractMatrix,input_field::AbstractMatrix,weights,indices)

    @parallel for i in axes(output_field,2)
        @inbounds begin
        inds = indices[i]
        w = weights[i]
        @simd for k in axes(output_field,1)
            output_field[k,i] = weighted_sum(input_field,w,inds,k)
        end
        end #inbounds
    end

    return output_field
end

function weighted_sum_transformation!(output_field::AbstractMatrix,input_field::AbstractMatrix,op::F,weights,indices) where F<:Function

    @parallel for i in axes(output_field,2)
        @inbounds begin
        inds = indices[i]
        w = weights[i]
        @simd for k in axes(output_field,1)
            output_field[k,i] = weighted_sum(op,input_field,w,inds,k)
        end
        end #inbounds
    end

    return output_field
end

function weighted_sum_transformation!(output_field::AbstractMatrix,op::F,input_field::AbstractMatrix,weights,indices) where F<:Function

    @parallel for i in axes(output_field,2)
        @inbounds begin
        inds = indices[i]
        w = weights[i]
        @simd for k in axes(output_field,1)
            output_field[k,i] = op(output_field[k,i],weighted_sum(input_field,w,inds,k))
        end
        end #inbounds
    end

    return output_field
end

function weighted_sum_transformation!(output_field::AbstractMatrix,op::F,input_field::AbstractMatrix,op2::F2,weights,indices) where {F<:Function,F2<:Function}

    @parallel for i in axes(output_field,2)
        @inbounds begin
        inds = indices[i]
        w = weights[i]
        @simd for k in axes(output_field,1)
            output_field[k,i] = op(output_field[k,i],weighted_sum(op2,input_field,w,inds,k))
        end
        end #inbounds 
    end

    return output_field
end

function weighted_sum_transformation!(output_field::AbstractArray{<:Any,3},input_field::AbstractArray{<:Any,3},weights,indices)

    @parallel for i in axes(output_field,2)
        @inbounds begin
        inds = indices[i]
        w = weights[i]
        for t in axes(output_field,3)
            @simd for k in axes(output_field,1)
                output_field[k,i,t] = weighted_sum(input_field,w,inds,k,t)
            end
        end
        end #inbounds
    end

    return output_field
end

function weighted_sum_transformation!(output_field::AbstractArray{<:Any,3},input_field::AbstractArray{<:Any,3},op::F,weights,indices) where {F<:Function}

    @parallel for i in axes(output_field,2)
        @inbounds begin
        inds = indices[i]
        w = weights[i]
        for t in axes(output_field,3)
            @simd for k in axes(output_field,1)
                output_field[k,i,t] = weighted_sum(op,input_field,w,inds,k,t)
            end
        end
        end #inbounds
    end

    return output_field
end

function weighted_sum_transformation!(output_field::AbstractArray{<:Any,3},op::F,input_field::AbstractArray{<:Any,3},weights,indices) where {F<:Function}

    @parallel for i in axes(output_field,2)
        @inbounds begin
        inds = indices[i]
        w = weights[i]
        for t in axes(output_field,3)
            @simd for k in axes(output_field,1)
                output_field[k,i,t] = op(output_field[k,i,t],weighted_sum(input_field,w,inds,k,t))
            end
        end
        end #inbounds 
    end

    return output_field
end

function weighted_sum_transformation!(output_field::AbstractArray{<:Any,3},op::F,input_field::AbstractArray{<:Any,3},op2::F2,weights,indices) where {F<:Function,F2<:Function}

    @parallel for i in axes(output_field,2)
        @inbounds begin
        inds = indices[i]
        w = weights[i]
        for t in axes(output_field,3)
            @simd for k in axes(output_field,1)
                output_field[k,i,t] = op(output_field[k,i,t],weighted_sum(op2,input_field,w,inds,k,t))
            end
        end
        end #inbounds
    end

    return output_field
end
