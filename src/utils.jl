const IntOrVecRange = Union{Int, Int32, <:sd.VecRange}

# @inline get_node_index(i::Integer) = i
# @inline get_node_index(k::Integer, i::Integer) = i
# @inline get_node_index(k::Integer, i::Integer, t::Integer) = i
# @inline insert_index(inds_input::NTuple{0}, inds_output::NTuple{N, T}) where {N, T <: Integer} = map(Int, inds_output)
# @inline insert_index(inds_input::NTuple{1}, inds_output::NTuple{N, T}) where {N, T <: Integer} = ntuple(i -> (Int(inds_input[1]), Int(inds_output[i])), Val{N}())
# @inline insert_index(inds_input::NTuple{2}, inds_output::NTuple{N, T}) where {N, T <: Integer} = ntuple(i -> (Int(inds_input[1]), Int(inds_output[i]), Int(inds_input[2])), Val{N}())

@inline construct_new_node_index(i::Integer, n::Integer) = (oftype(i, n),)
@inline construct_new_node_index(k::Integer, i::Integer, n::Integer) = (k, oftype(i, n))
@inline construct_new_node_index(k::Integer, i::Integer, t::Integer, n::Integer) = (k, oftype(i, n), t)

@inline is_proper_size(field::AbstractVector, n::Integer) = length(field) == n
@inline is_proper_size(field::AbstractMatrix, n::Integer) = size(field, 2) == n
@inline is_proper_size(field::AbstractArray{<:Any, 3}, n::Integer) = size(field, 2) == n

@inline simd_length(::Type{T}) where {T} = 64 ÷ sizeof(T)

@inline simd_repeat(t::Val, v::Number) = sd.Vec(ntuple(i -> v, t)...)
@inline simd_repeat(t::Val, v::NTuple{N}) where {N} = map(simd_repeat, ntuple(i -> t, Val{N}()), v)
@inline simd_repeat(t::Val, v::ImmutableVector{N}) where {N} = map(simd_repeat, (@inbounds ImmutableVector(ntuple(i -> t, Val{N}()), v.length)), v)
@inline simd_repeat(t::Val, v::Vec2Dxy) = Vec(x = sd.Vec(ntuple(i -> v.x, t)...), y = sd.Vec(ntuple(i -> v.y, t)...))
@inline simd_repeat(t::Val, v::Vec3D) = Vec(x = sd.Vec(ntuple(i -> v.x, t)...), y = sd.Vec(ntuple(i -> v.y, t)...), z = sd.Vec(ntuple(i -> v.z, t)...))

@inline function simd_ranges(N, len)
    if len < N
        simd_range = 1:1:0
        serial_range = 1:len
    elseif rem(len, N) == 0
        simd_range = 1:N:len
        serial_range = 1:0
    else
        simd_range = 1:N:(len - N)
        serial_range = (length(simd_range) * N + 1):len
    end
    return simd_range, serial_range
end

@inline function mean_sum(input_field, ind::NTuple{N, T}, op::F = Base.identity) where {F <: Function, T <: Integer, N}
    mapreduce(@inline(x -> @inline(op(@inbounds(input_field[x])))), +, ind) / N
end

@inline function mean_sum(input_field, ind::NTuple{N, T}, k::IntOrVecRange, op::F = Base.identity) where {F <: Function, T <: Integer, N}
    mapreduce(@inline(x -> @inline(op(@inbounds(input_field[k, x])))), +, ind) / N
end

@inline function mean_sum(input_field, ind::NTuple{N, T}, k::IntOrVecRange, t::Integer, op::F = Base.identity) where {F <: Function, T <: Integer, N}
    mapreduce(@inline(x -> @inline(op(@inbounds(input_field[k, x, t])))), +, ind) / N
end

function to_mean_transformation!(output_field::AbstractVector, input_field::AbstractVector, indices, op::F = Base.identity) where {F <: Function}
    @parallel for i in eachindex(output_field)
        @inbounds output_field[i] = mean_sum(input_field, indices[i], op)
    end
    return output_field
end

function to_mean_transformation!(output_field::AbstractVector, op_out::F, input_field::AbstractVector, indices, op::FI = Base.identity) where {F <: Function, FI <: Function}
    @parallel for i in eachindex(output_field)
        @inbounds output_field[i] = @inline op_out(output_field[i], mean_sum(input_field, indices[i], op))
    end
    return output_field
end

function to_mean_transformation!(output_field::AbstractMatrix{T}, input_field::AbstractMatrix, indices, op::F = Base.identity) where {T, F <: Function}

    N_SIMD = simd_length(T)
    lane = sd.VecRange{N_SIMD}(0)
    Nk = size(output_field, 1)

    range_simd, range_serial = simd_ranges(N_SIMD, Nk)

    @parallel for i in axes(output_field, 2)
        @inbounds begin
            inds = map(Int, indices[i])

            for k in range_simd
                k_simd = lane + k
                output_field[k_simd, i] = mean_sum(input_field, inds, k_simd, op)
            end

            for k in range_serial
                output_field[k, i] = mean_sum(input_field, inds, k, op)
            end

        end #inbounds
    end

    return output_field
end

function to_mean_transformation!(output_field::AbstractMatrix{T}, op_out::F, input_field::AbstractMatrix, indices, op::F2 = Base.identity) where {T, F <: Function, F2 <: Function}

    N_SIMD = simd_length(T)
    lane = sd.VecRange{N_SIMD}(0)
    Nk = size(output_field, 1)

    range_simd, range_serial = simd_ranges(N_SIMD, Nk)

    @parallel for i in axes(output_field, 2)
        @inbounds begin
            inds = map(Int, indices[i])

            for k in range_simd
                k_simd = lane + k
                output_field[k_simd, i] = @inline op_out(output_field[k_simd, i], mean_sum(input_field, inds, k_simd, op))
            end

            for k in range_serial
                output_field[k, i] = @inline op_out(output_field[k, i], mean_sum(input_field, inds, k, op))
            end

        end #inbounds
    end

    return output_field
end

function to_mean_transformation!(output_field::AbstractArray{T, 3}, input_field::AbstractArray{<:Any, 3}, indices, op::F = Base.identity) where {T, F <: Function}

    N_SIMD = simd_length(T)
    lane = sd.VecRange{N_SIMD}(0)
    Nk = size(output_field, 1)

    range_simd, range_serial = simd_ranges(N_SIMD, Nk)

    @parallel for i in axes(output_field, 2)
        @inbounds begin
            inds = map(Int, indices[i])

            for t in axes(output_field, 3)
                for k in range_simd
                    k_simd = lane + k
                    output_field[k_simd, i, t] = mean_sum(input_field, inds, k_simd, t, op)
                end

                for k in range_serial
                    output_field[k, i, t] = mean_sum(input_field, inds, k, t, op)
                end
            end

        end #inbounds
    end

    return output_field
end

function to_mean_transformation!(output_field::AbstractArray{T, 3}, op_out::F, input_field::AbstractArray{<:Any, 3}, indices, op::F2 = Base.identity) where {T, F <: Function, F2 <: Function}

    N_SIMD = simd_length(T)
    lane = sd.VecRange{N_SIMD}(0)
    Nk = size(output_field, 1)

    range_simd, range_serial = simd_ranges(N_SIMD, Nk)

    @parallel for i in axes(output_field, 2)
        @inbounds begin
            inds = map(Int, indices[i])

            for t in axes(output_field, 3)
                for k in range_simd
                    k_simd = lane + k
                    output_field[k_simd, i, t] = @inline op_out(output_field[k_simd, i, t], mean_sum(input_field, inds, k_simd, t, op))
                end

                for k in range_serial
                    output_field[k, i, t] = @inline op_out(output_field[k, i, t], mean_sum(input_field, inds, k, t, op))
                end
            end

        end #inbounds
    end

    return output_field
end

#@inline function dropfirst(t::NTuple{N}) where {N}
#    return ntuple(i -> t[i + 1], Val{N - 1}())
#end

@inline function droplast(t::NTuple{N}) where {N}
    return ntuple(@inline(i -> @inbounds(t[i])), Val{N - 1}())
end

@inbounds @inline function weighted_sum(input_field::AbstractVector, weights::NTuple{2}, indices::NTuple{2}, op::F = Base.identity) where {F <: Function}
    muladd(weights[2], @inline(op(input_field[indices[2]])), weights[1] * @inline(op(input_field[indices[1]])))
end

@inbounds @inline function weighted_sum(input_field::AbstractMatrix, weights::NTuple{2}, indices::NTuple{2}, k::IntOrVecRange, op::F = Base.identity) where {F <: Function}
    muladd(weights[2], @inline(op(input_field[k, indices[2]])), weights[1] * @inline(op(input_field[k, indices[1]])))
end

@inbounds @inline function weighted_sum(input_field::AbstractArray{<:Any, 3}, weights::NTuple{2}, indices::NTuple{2}, k::IntOrVecRange, t::Integer, op::F = Base.identity) where {F <: Function}
    muladd(weights[2], @inline(op(input_field[k, indices[2], t])), weights[1] * @inline(op(input_field[k, indices[1], t])))
end

@inbounds @inline function weighted_sum(input_field::AbstractVector, weights::NTuple{N}, indices::NTuple{N}, op::F = Base.identity) where {N, F <: Function}
    @inbounds muladd(weights[N], @inline(op(input_field[indices[N]])), weighted_sum(input_field, droplast(weights), droplast(indices), op))
end

@inbounds @inline function weighted_sum(input_field::AbstractMatrix, weights::NTuple{N}, indices::NTuple{N}, k::IntOrVecRange, op::F = Base.identity) where {N, F <: Function}
    @inbounds muladd(weights[N], @inline(op(input_field[k, indices[N]])), weighted_sum(input_field, droplast(weights), droplast(indices), k, op))
end

@inbounds @inline function weighted_sum(input_field::AbstractArray{<:Any, 3}, weights::NTuple{N}, indices::NTuple{N}, k::IntOrVecRange, t::Integer, op::F = Base.identity) where {N, F <: Function}
    @inbounds muladd(weights[N], @inline(op(input_field[k, indices[N], t])), weighted_sum(input_field, droplast(weights), droplast(indices), k, t, op))
end

@inline function weighted_sum(input_field::AbstractVector, weights, indices, op::F = Base.identity) where {F <: Function}
    r = zero(Base.promote_op(*, eltype(weights), Base.promote_op(op, eltype(input_field))))
    @inbounds for i in eachindex(indices)
        ind = indices[i]
        r = muladd(weights[i], @inline(op(input_field[ind])), r)
    end
    return r
end

@inline function weighted_sum(input_field::AbstractMatrix, weights, indices, k::IntOrVecRange, op::F = Base.identity) where {F <: Function}
    r = zero(Base.promote_op(*, eltype(weights), Base.promote_op(op, eltype(input_field))))
    @inbounds for i in eachindex(indices)
        ind = indices[i]
        r = muladd(weights[i], @inline(op(input_field[k, ind])), r)
    end
    return r
end

@inline function weighted_sum(input_field::AbstractArray{<:Any, 3}, weights, indices, k::IntOrVecRange, t::Integer, op::F = Base.identity) where {F <: Function}
    r = zero(Base.promote_op(*, eltype(weights), Base.promote_op(op, eltype(input_field))))
    @inbounds for i in eachindex(indices)
        ind = indices[i]
        r = muladd(weights[i], @inline(op(input_field[k, ind, t])), r)
    end
    return r
end

function weighted_sum_transformation!(output_field::AbstractVector, input_field::AbstractVector, weights, indices, op::F = Base.identity) where {F <: Function}
    @parallel for i in eachindex(output_field)
        @inbounds output_field[i] = @inline weighted_sum(input_field, weights[i], indices[i], op)
    end
    return output_field
end

function weighted_sum_transformation!(output_field::AbstractVector, op_out::F, input_field::AbstractVector, weights, indices, op::FI = Base.identity) where {F <: Function, FI <: Function}
    @parallel for i in eachindex(output_field)
        @inbounds output_field[i] = @inline op_out(output_field[i], @inline weighted_sum(input_field, weights[i], indices[i], op))
    end
    return output_field
end

function weighted_sum_transformation!(output_field::AbstractMatrix{T}, input_field::AbstractMatrix, weights, indices, op::F = Base.identity) where {T, F <: Function}

    N_SIMD = simd_length(T)
    ValN_SIMD = Val{N_SIMD}()
    lane = sd.VecRange{N_SIMD}(0)
    Nk = size(output_field, 1)

    range_simd, range_serial = simd_ranges(N_SIMD, Nk)

    @parallel for i in axes(output_field, 2)
        @inbounds begin
            inds = map(Int, indices[i])
            w = weights[i]
            w_simd = simd_repeat(ValN_SIMD, w)

            for k in range_simd
                k_simd = lane + k
                output_field[k_simd, i] = @inline weighted_sum(input_field, w_simd, inds, k_simd, op)
            end

            for k in range_serial
                output_field[k, i] = @inline weighted_sum(input_field, w, inds, k, op)
            end

        end #inbounds
    end

    return output_field
end

function weighted_sum_transformation!(output_field::AbstractMatrix{T}, op_out::F, input_field::AbstractMatrix, weights, indices, op::F2 = Base.identity) where {T, F <: Function, F2 <: Function}

    N_SIMD = simd_length(T)
    ValN_SIMD = Val{N_SIMD}()
    lane = sd.VecRange{N_SIMD}(0)
    Nk = size(output_field, 1)

    range_simd, range_serial = simd_ranges(N_SIMD, Nk)

    @parallel for i in axes(output_field, 2)
        @inbounds begin
            inds = map(Int, indices[i])
            w = weights[i]
            w_simd = simd_repeat(ValN_SIMD, w)

            for k in range_simd
                k_simd = lane + k
                output_field[k_simd, i] = @inline op_out(output_field[k_simd, i], @inline weighted_sum(input_field, w_simd, inds, k_simd, op))
            end

            for k in range_serial
                output_field[k, i] = @inline op_out(output_field[k, i], @inline weighted_sum(input_field, w, inds, k, op))
            end

        end #inbounds
    end

    return output_field
end

function weighted_sum_transformation!(output_field::AbstractArray{T, 3}, input_field::AbstractArray{<:Any, 3}, weights, indices, op::F = Base.identity) where {T, F <: Function}

    N_SIMD = simd_length(T)
    ValN_SIMD = Val{N_SIMD}()
    lane = sd.VecRange{N_SIMD}(0)
    Nk = size(output_field, 1)

    range_simd, range_serial = simd_ranges(N_SIMD, Nk)

    @parallel for i in axes(output_field, 2)
        @inbounds begin
            inds = map(Int, indices[i])
            w = weights[i]
            w_simd = simd_repeat(ValN_SIMD, w)

            for t in axes(output_field, 3)
                for k in range_simd
                    k_simd = lane + k
                    output_field[k_simd, i, t] = @inline weighted_sum(input_field, w_simd, inds, k_simd, t, op)
                end

                for k in range_serial
                    output_field[k, i, t] = @inline weighted_sum(input_field, w, inds, k, t, op)
                end
            end

        end #inbounds
    end

    return output_field
end

function weighted_sum_transformation!(output_field::AbstractArray{T, 3}, op_out::F, input_field::AbstractArray{<:Any, 3}, weights, indices, op::F2 = Base.identity) where {T, F <: Function, F2 <: Function}

    N_SIMD = simd_length(T)
    ValN_SIMD = Val{N_SIMD}()
    lane = sd.VecRange{N_SIMD}(0)
    Nk = size(output_field, 1)

    range_simd, range_serial = simd_ranges(N_SIMD, Nk)

    @parallel for i in axes(output_field, 2)
        @inbounds begin
            inds = map(Int, indices[i])
            w = weights[i]
            w_simd = simd_repeat(ValN_SIMD, w)

            for t in axes(output_field, 3)
                for k in range_simd
                    k_simd = lane + k
                    output_field[k_simd, i, t] = @inline op_out(output_field[k_simd, i, t], @inline weighted_sum(input_field, w_simd, inds, k_simd, t, op))
                end

                for k in range_serial
                    output_field[k, i, t] = @inline op_out(output_field[k, i, t], @inline weighted_sum(input_field, w, inds, k, t, op))
                end
            end

        end #inbounds
    end

    return output_field
end
