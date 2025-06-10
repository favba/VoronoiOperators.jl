@inline function compute_weights_lsq2(bpos::VecND{TF}, elements_pos, 𝐢::Vec, 𝐣::Vec) where {TF}

    nElem = length(elements_pos)

    M = Matrix{TF}(undef, nElem, 3)

    for eli in Base.OneTo(nElem)
        M[eli, 1] = oneunit(TF)
        elpos = elements_pos[eli]
        dpos = elpos - bpos
        M[eli, 2] = dpos ⋅ 𝐢
        M[eli, 3] = dpos ⋅ 𝐣
    end

    MpM = cholesky!(Hermitian(M'*M))

    wvec = vec((LinearAlgebra.inv!(MpM)*M')[1,:])

    return wvec
end

@inline function compute_weights_lsq3(bpos::VecND{TF}, elements_pos, 𝐢::Vec, 𝐣::Vec) where {TF}

    nElem = length(elements_pos)

    M = Matrix{TF}(undef, nElem, 6)

    for eli in Base.OneTo(nElem)
        M[eli, 1] = oneunit(TF)
        elpos = elements_pos[eli]
        dpos = elpos - bpos
        xterm = dpos ⋅ 𝐢
        yterm = dpos ⋅ 𝐣
        M[eli, 2] = xterm
        M[eli, 3] = yterm
        M[eli, 4] = xterm^2
        M[eli, 5] = yterm^2
        M[eli, 6] = xterm*yterm
    end

    MpM = Hermitian(M'*M)

    #Perform regularization if condition number is too big (or Inf)
    #Regularize only quadratic terms, to preserve 1st order and 2nd order terms
    cn = cond(MpM)
    ii = 0
    while cn > 5e6
        ii += 1
        #@show c, ii, cn
        for i in 4:6
            MpM[i, i] += 1e-6
        end
        cn = cond(MpM)
    end

    wvec = vec((LinearAlgebra.inv!(cholesky!(MpM))*M')[1,:])

    return wvec
end

@inline _data(w) = w
@inline _data(w::ImVecArray) = w.data

@inline _pad_with_zero(::Type{ImmutableVector{NE, TF}}, wvec) where {NE, TF} = padwith(ImmutableVector{NE, TF}(wvec), zero(TF)).data
@inline _pad_with_zero(::Type{NTuple{N, TF}}, wvec) where {N, TF} = wvec

function compute_weights_lsq_periodic!(w::AbstractVector{TV}, basePos, elemPos, elemOnBase, xp::Number, yp::Number, lsq_func::F) where {TV, F}

    wdata = _data(w)

    @parallel for b in eachindex(basePos)
        @inbounds begin

            bpos = basePos[b]

            elements = map(@inline(i -> closest(bpos, @inbounds(elemPos[i]), xp, yp)), elemOnBase[b])

            @inline wvec = lsq_func(bpos, elements, 𝐢, 𝐣)

            wdata[b] = _pad_with_zero(TV, wvec)
        end
    end

    return w
end

function compute_weights_lsq_spherical!(w::AbstractVector{TV}, basePos, elemPos, elemOnBase, R::Number, lsq_func::F) where {TV, F}

    wdata = _data(w)

    @parallel for b in eachindex(basePos)
        @inbounds begin

            bpos_n = basePos[b] / R

            elements = map(@inline(i -> @inbounds(elemPos[i]) / R), elemOnBase[b])
            elements_proj = project_points_to_tangent_plane(1, bpos_n, elements)

            east_v = eastward_vector(bpos_n)
            north_v = bpos_n × east_v
            @inline wvec = lsq_func(bpos_n, elements_proj, east_v, north_v)

            wdata[b] = _pad_with_zero(TV, wvec)
        end
    end

    return w
end

function build_weight_vector(pos::VecArray, elemOnBase::ImVecArray{NE, TI, 1}) where {NE, TI}
    TF = nonzero_eltype(eltype(pos))
    w = ImmutableVectorArray(Vector{NTuple{NE,TF}}(undef, length(pos)), elemOnBase.length)
    return w
end

function build_weight_vector(pos::VecArray, ::AbstractVector{NTuple{NE, TI}}) where {NE, TI}
    TF = nonzero_eltype(eltype(pos))
    w = Vector{NTuple{NE,TF}}(undef, length(pos))
    return w
end

function compute_weights_lsq(basePos, elemPos, elemOnBase, xp::Number, yp::Number, lsq_func::F) where {F}
    w = build_weight_vector(basePos, elemOnBase)
    return compute_weights_lsq_periodic!(w, basePos, elemPos, elemOnBase, xp ,yp, lsq_func)
end

function compute_weights_lsq(basePos, elemPos, elemOnBase, R::Number, lsq_func::F) where {F}
    w = build_weight_vector(basePos, elemOnBase)
    return compute_weights_lsq_spherical!(w, basePos, elemPos, elemOnBase, R, lsq_func)
end

