_capacity(::Type{<:FixedVector{N}}) where {N} = N
_capacity(::Type{<:SmallVector{N}}) where {N} = N
_capacity(a) = _capacity(typeof(a))

@inline function compute_weights_lsq2(bpos::VecND{TF}, elements_pos, 𝐢::Vec, 𝐣::Vec) where {TF}
    @inbounds begin
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
    P = LinearAlgebra.inv!(MpM)*M'

    N = _capacity(elements_pos)
    wvec = map(x -> @inbounds(P[1, x]), SmallVector{N}(Base.OneTo(nElem)))
    end #inbounds

    return wvec
end

@inline function compute_weights_lsq3(bpos::VecND{TF}, elements_pos, 𝐢::Vec, 𝐣::Vec) where {TF}
    @inbounds begin
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

    P = LinearAlgebra.inv!(cholesky!(MpM))*M'
    N = _capacity(elements_pos)
    wvec = map(x -> @inbounds(P[1, x]), SmallVector{N}(Base.OneTo(nElem)))
    end #inbounds

    return wvec
end

@inline _data(w) = w
@inline _data(w::SmVecArray) = w.data

@inline _get_data(::Type{SmallVector{NE, TF}}, wvec) where {NE, TF} = fixedvector(wvec)
@inline _get_data(::Type{FixedVector{N, TF}}, wvec) where {N, TF} = wvec

function compute_weights_lsq_periodic!(w::AbstractVector{TV}, basePos, elemPos, elemOnBase, xp::Number, yp::Number, lsq_func::F) where {TV, F}

    wdata = _data(w)

    @parallel for b in eachindex(basePos)
        @inbounds begin

            bpos = basePos[b]

            elements = map(@inline(i -> closest(bpos, @inbounds(elemPos[i]), xp, yp)), elemOnBase[b])

            @inline wvec = lsq_func(bpos, elements, 𝐢, 𝐣)

            wdata[b] = _get_data(TV, wvec)
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

            wdata[b] = _get_data(TV, wvec)
        end
    end

    return w
end

function build_weight_vector(pos::VecArray, elemOnBase::SmVecArray{NE, TI, 1}) where {NE, TI}
    TF = nonzero_eltype(eltype(pos))
    w = SmallVectorArray(Vector{FixedVector{NE,TF}}(undef, length(pos)), elemOnBase.length)
    return w
end

function build_weight_vector(pos::VecArray, ::AbstractVector{FixedVector{NE, TI}}) where {NE, TI}
    TF = nonzero_eltype(eltype(pos))
    w = Vector{FixedVector{NE,TF}}(undef, length(pos))
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

@inline function compute_weights_vec_lsq1(elements_normal, 𝐢::Vec, 𝐣::Vec)
    @inbounds begin
    nElem = length(elements_normal)

    TF = nonzero_eltype(eltype(elements_normal))
    M = Matrix{TF}(undef, nElem, 2)

    for eli in Base.OneTo(nElem)
        normal = elements_normal[eli]
        M[eli, 1] = normal ⋅ 𝐢
        M[eli, 2] = normal ⋅ 𝐣
    end

    MpM = cholesky!(Hermitian(M'*M))

    P = LinearAlgebra.inv!(MpM)*M'
    N = _capacity(elements_normal)
    wvec = map(x -> (@inbounds(P[1, x])*𝐢 + @inbounds(P[2, x])*𝐣), SmallVector{N}(Base.OneTo(nElem)))
    end #inbounds

    return wvec
end

function compute_weights_vec_lsq1_periodic!(w::AbstractVector{TV}, elemOnBase, elemNormals) where {TV}

    wdata = _data(w)

    @parallel for b in eachindex(elemOnBase)
        @inbounds begin

            elements_normals = map(@inline(i -> @inbounds(elemNormals[i])), elemOnBase[b])

            @inline wvec = compute_weights_vec_lsq1(elements_normals, 𝐢, 𝐣)

            wdata[b] = _get_data(TV, wvec)
        end
    end

    return w
end

function compute_weights_vec_lsq1_spherical!(w::AbstractVector{TV}, basePos, elemOnBase, elemNormals, R::Number) where {TV}

    wdata = _data(w)

    @parallel for b in eachindex(basePos)
        @inbounds begin

            bpos_n = basePos[b] / R

            elements_normal_proj = map(elemOnBase[b]) do i
                @inline
                _n = @inbounds(elemNormals[i])
                return normalize(_n - (_n ⋅ bpos_n)*bpos_n)
            end

            east_v = eastward_vector(bpos_n)
            north_v = bpos_n × east_v
            @inline wvec = compute_weights_vec_lsq1(elements_normal_proj, east_v, north_v)

            wdata[b] = _get_data(TV, wvec)
        end
    end

    return w
end

@inline function compute_weights_vec_lsq2(bpos::VecND{TF}, elements_pos, elements_normal, 𝐢::Vec, 𝐣::Vec) where {TF}
    @inbounds begin
    nElem = length(elements_normal)

    M = Matrix{TF}(undef, nElem, 6)

    for eli in Base.OneTo(nElem)
        normal = elements_normal[eli]
        nx = normal ⋅ 𝐢
        ny = normal ⋅ 𝐣
        dp = elements_pos[eli] - bpos
        dpx = dp ⋅ 𝐢
        dpy = dp ⋅ 𝐣
        M[eli, 1] = nx
        M[eli, 2] = ny
        M[eli, 3] = nx*dpx
        M[eli, 4] = ny*dpx
        M[eli, 5] = nx*dpy
        M[eli, 6] = ny*dpy
    end

    MpM = Hermitian(M'*M)
    cn = cond(MpM)
    ii = 0
    while cn > 5e6
        ii += 1
        for i in 3:6
            MpM[i, i] += 1e-6
        end
        cn = cond(MpM)
    end

    P = LinearAlgebra.inv!(cholesky!(MpM))*M'

    N = _capacity(elements_normal)
    wvec = map(x -> (@inbounds(P[1, x])*𝐢 + @inbounds(P[2, x])*𝐣), SmallVector{N}(Base.OneTo(nElem)))
    end #inbounds

    return wvec
end

function compute_weights_vec_lsq2_periodic!(w::AbstractVector{TV}, basePos, elemOnBase, elemPos, elemNormals, xp::Number, yp::Number) where {TV}

    wdata = _data(w)

    @parallel for b in eachindex(basePos)
        @inbounds begin

            bpos = basePos[b]

            eob = elemOnBase[b]

            elements = map(@inline(i -> closest(bpos, @inbounds(elemPos[i]), xp, yp)), eob)
            elements_normal = map(@inline(i -> @inbounds(elemNormals[i])), eob)

            @inline wvec = compute_weights_vec_lsq2(bpos, elements, elements_normal, 𝐢, 𝐣)

            wdata[b] = _get_data(TV, wvec)
        end
    end

    return w
end

function compute_weights_vec_lsq2_spherical!(w::AbstractVector{TV}, basePos, elemOnBase, elemPos, elemNormals, R::Number) where {TV}

    wdata = _data(w)

    @parallel for b in eachindex(basePos)
        @inbounds begin

            bpos_n = basePos[b] / R

            eob = elemOnBase[b]

            elements = map(@inline(i -> @inbounds(elemPos[i]) / R), eob)
            elements_proj = project_points_to_tangent_plane(1, bpos_n, elements)

            elements_normal_proj = map(eob) do i
                @inline
                _n = @inbounds(elemNormals[i])
                return normalize(_n - (_n ⋅ bpos_n)*bpos_n)
            end

            east_v = eastward_vector(bpos_n)
            north_v = bpos_n × east_v
            @inline wvec = compute_weights_vec_lsq2(bpos_n, elements_proj, elements_normal_proj, east_v, north_v)

            wdata[b] = _get_data(TV, wvec)
        end
    end

    return w
end
