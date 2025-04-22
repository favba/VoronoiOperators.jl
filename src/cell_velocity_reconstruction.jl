abstract type CellVelocityReconstruction{N_MAX, TI, TF, TZ} <: LinearVoronoiOperator end
name_input(::CellVelocityReconstruction) = "edge"
name_output(::CellVelocityReconstruction) = "cell"

struct CellVelocityReconstructionPerot{N_MAX, TI, TF, TZ} <: CellVelocityReconstruction{N_MAX, TI, TF, TZ}
    n::Int
    indices::ImVecArray{N_MAX, TI, 1}
    weights::ImVecArray{N_MAX, Vec{Union{TF, TZ}, 1, TF, TF, TZ}, 1}
end

function compute_weights_perot_velocity_reconstruction_periodic!(w::AbstractVector{ImmutableVector{N_MAX, T}}, c_pos, aC, Le, ne, edgesOnCell::AbstractVector{<:ImmutableVector{N_MAX}}, v_pos, verticesOnEdge, xp::Number, yp::Number) where {T, N_MAX}

    wdata = w.data
    @parallel for c in eachindex(edgesOnCell)
        @inbounds begin
        cp = c_pos[c]
        inv_a = inv(aC[c])

        eoc = edgesOnCell[c]
        l = length(eoc)
        aux = ImmutableVector{N_MAX,T}()

        for i in Base.OneTo(l)
            e = eoc[i]
            v1, v2 = verticesOnEdge[e]
            v1p = closest(cp, v_pos[v1], xp, yp)
            v2p = closest(cp, v_pos[v2], xp, yp)
            ep = (v1p + v2p) / 2
            r_vec = cp - ep
            r_vec = sign(ne[e] ⋅ r_vec) * r_vec
            aux = @inbounds push(aux, inv_a * r_vec * Le[e])
        end
        wdata[c] = padwith(aux, zero(T)).data
        end #inbounds
    end
    return w
end

function CellVelocityReconstructionPerot(cells::Cells{false, N_MAX, TI, TF}, edges::Edges, vertices::Vertices, x_period::Number, y_period::Number) where {N_MAX, TI, TF}
    edgesOnCell = cells.edges
    weights =ImmutableVectorArray(Vector{NTuple{N_MAX, Vec2Dxy{TF}}}(undef, cells.n), edgesOnCell.length)
    compute_weights_perot_velocity_reconstruction_periodic!(weights, cells.position, cells.area, edges.length, edges.normal, edgesOnCell, vertices.position, edges.vertices, x_period, y_period)
    return CellVelocityReconstructionPerot(edges.n, edgesOnCell, weights)
end

function CellVelocityReconstructionPerot(mesh::AbstractVoronoiMesh{false})
    CellVelocityReconstructionPerot(mesh.cells, mesh.edges, mesh.vertices, mesh.x_period, mesh.y_period)
end

function compute_weights_perot_velocity_reconstruction_spherical!(w::AbstractVector{ImmutableVector{N_MAX, T}}, R::Number, c_pos, aC, Le, ne, edgesOnCell::AbstractVector{<:ImmutableVector{N_MAX}}, v_pos, verticesOnEdge) where {T, N_MAX}

    wdata = w.data
    @parallel for c in eachindex(edgesOnCell)
        @inbounds begin
        cp_p = c_pos[c]
        cp_n = cp_p / R

        inv_a = inv(aC[c])

        eoc = edgesOnCell[c]
        l = length(eoc)
        aux = ImmutableVector{N_MAX,T}()

        for i in Base.OneTo(l)
            e = eoc[i]
            v1, v2 = verticesOnEdge[e]
            v1p = v_pos[v1]
            v2p = v_pos[v2]
            ep_n = normalize((v1p + v2p) / 2)
            ep = ep_n * (R/(ep_n ⋅ cp_n)) # Edge midpoint projection on the plane tangent to the cell Voronoi generator point

            r_vec = cp_p - ep
            r_vec = sign(ne[e] ⋅ r_vec) * r_vec
            aux = @inbounds push(aux, inv_a * r_vec * Le[e])
        end
        wdata[c] = padwith(aux, zero(T)).data
        end #inbounds
    end
    return w
end

function CellVelocityReconstructionPerot(cells::Cells{true, N_MAX, TI, TF}, edges::Edges{true}, vertices::Vertices{true}) where {N_MAX, TI, TF}
    edgesOnCell = cells.edges
    weights =ImmutableVectorArray(Vector{NTuple{N_MAX, Vec3D{TF}}}(undef, cells.n), edgesOnCell.length)
    compute_weights_perot_velocity_reconstruction_spherical!(weights, cells.sphere_radius, cells.position, cells.area, edges.length, edges.normal, edgesOnCell, vertices.position, edges.vertices)
    return CellVelocityReconstructionPerot(edges.n, edgesOnCell, weights)
end

function CellVelocityReconstructionPerot(mesh::AbstractVoronoiMesh{true})
    CellVelocityReconstructionPerot(mesh.cells, mesh.edges, mesh.vertices)
end

struct CellVelocityReconstructionLSq1{N_MAX, TI, TF, TZ} <: CellVelocityReconstruction{N_MAX, TI, TF, TZ}
    n::Int
    indices::ImVecArray{N_MAX, TI, 1}
    weights::ImVecArray{N_MAX, Vec{Union{TF, TZ}, 1, TF, TF, TZ}, 1}
end

function compute_weights_interp_velocity_reconstruction_periodic!(w::AbstractVector{<:ImmutableVector{N_MAX, Vec2Dxy{TF}}}, ne, edgesOnCell::AbstractVector{<:ImmutableVector{N_MAX}}) where {TF, N_MAX}

    wdata = w.data
    @parallel for c in eachindex(edgesOnCell)
        @inbounds begin

        eoc = edgesOnCell[c]
        l = length(eoc)

        M = Matrix{TF}(undef, l, 2)
        for i in Base.OneTo(l)
            e = eoc[i]
            n = ne[e]
            M[i, 1] = n.x
            M[i, 2] = n.y
        end
        MpM = cholesky!(Hermitian(M'*M))
        P = LinearAlgebra.inv!(MpM)*M'

        wa = map(x -> Vec(x = @inbounds(P[1, x]), y = @inbounds(P[2, x])), Base.OneTo(l))
        w = ImmutableVector{N_MAX}(wa)

        wdata[c] = padwith(w, zero(Vec2Dxy{TF})).data
        end #inbounds
    end
    return w
end

function CellVelocityReconstructionLSq1(cells::Cells{false, N_MAX, TI, TF}, edges::Edges) where {N_MAX, TI, TF}
    edgesOnCell = cells.edges
    weights =ImmutableVectorArray(Vector{NTuple{N_MAX, Vec2Dxy{TF}}}(undef, cells.n), edgesOnCell.length)
    compute_weights_interp_velocity_reconstruction_periodic!(weights, edges.normal, edgesOnCell)
    return CellVelocityReconstructionLSq1(edges.n, edgesOnCell, weights)
end

function compute_weights_interp_velocity_reconstruction_spherical!(w::AbstractVector{<:ImmutableVector{N_MAX, Vec3D{TF}}}, ne, cpos, edgesOnCell::AbstractVector{<:ImmutableVector{N_MAX}}, R::Number) where {TF, N_MAX}

    wdata = w.data
    @parallel for c in eachindex(edgesOnCell)
        @inbounds begin

        eoc = edgesOnCell[c]
        l = length(eoc)

        cp = cpos[c] / R

        xdir = zero(Vec3D{TF})
        ydir = zero(Vec3D{TF})

        M = Matrix{TF}(undef, l, 2)
        for i in Base.OneTo(l)
            e = eoc[i]
            _n = ne[e]
            n = normalize(_n - (_n ⋅ cp)*cp)
            if i == 1
                xdir = n
                ydir = normalize(cp × n)
                M[i, 1] = oneunit(TF)
                M[i, 2] = zero(TF)
            else
                M[i, 1] = n ⋅ xdir
                M[i, 2] = n ⋅ ydir
            end
        end
        MpM = cholesky!(Hermitian(M'*M))
        P = LinearAlgebra.inv!(MpM)*M'

        wa = map(x -> @inbounds(P[1,x])*xdir + @inbounds(P[2,x])*ydir, Base.OneTo(l))
        w = ImmutableVector{N_MAX}(wa)

        wdata[c] = padwith(w, zero(Vec3D{TF})).data
        end #inbounds
    end
    return w
end

function CellVelocityReconstructionLSq1(cells::Cells{true, N_MAX, TI, TF}, edges::Edges) where {N_MAX, TI, TF}
    edgesOnCell = cells.edges
    weights =ImmutableVectorArray(Vector{NTuple{N_MAX, Vec3D{TF}}}(undef, cells.n), edgesOnCell.length)
    compute_weights_interp_velocity_reconstruction_spherical!(weights, edges.normal, cells.position, edgesOnCell, cells.sphere_radius)
    return CellVelocityReconstructionLSq1(edges.n, edgesOnCell, weights)
end

function CellVelocityReconstructionLSq1(mesh::AbstractVoronoiMesh)
    CellVelocityReconstructionLSq1(mesh.cells, mesh.edges)
end

struct CellVelocityReconstructionLSq2{N_MAX, TI, TF, TZ} <: CellVelocityReconstruction{N_MAX, TI, TF, TZ}
    n::Int
    indices::ImVecArray{N_MAX, TI, 1}
    weights::ImVecArray{N_MAX, Vec{Union{TF, TZ}, 1, TF, TF, TZ}, 1}
end

function compute_weights_lsq2_velocity_reconstruction_periodic!(w::AbstractVector{<:ImmutableVector{N_MAX, Vec2Dxy{TF}}}, cpos, epos, ne, edgesOnCell::AbstractVector{<:ImmutableVector{N_MAX}}, xp::Number, yp::Number) where {TF, N_MAX}

    wdata = w.data
    @parallel for c in eachindex(edgesOnCell)
        @inbounds begin

        eoc = edgesOnCell[c]

        cp = cpos[c]

        l = length(eoc)

        M = Matrix{TF}(undef, l, 6)
        for i in Base.OneTo(l)
            e = eoc[i]
            n = ne[e]
            M[i, 1] = n.x
            M[i, 2] = n.y

            ep = closest(cp, epos[e], xp, yp)
            dp = ep - cp
            M[i, 3] = n.x*dp.x
            M[i, 4] = n.y*dp.x
            M[i, 5] = n.x*dp.y
            M[i, 6] = n.y*dp.y
        end

        MpM = Hermitian(M'*M)
        cn = cond(MpM)
        ii = 0
        while cn > 5e6
            ii += 1
            #@show c, ii, cn
            for i in 3:6
                MpM[i, i] += 1e-6
            end
            cn = cond(MpM)
        end

        P = LinearAlgebra.inv!(cholesky!(MpM))*M'

        wa = map(x -> Vec(x = @inbounds(P[1, x]), y = @inbounds(P[2, x])), Base.OneTo(l))
        w = ImmutableVector{N_MAX}(wa)

        wdata[c] = padwith(w, zero(Vec2Dxy{TF})).data
        end #inbounds
    end
    return w
end

function CellVelocityReconstructionLSq2(cells::Cells{false, N_MAX, TI, TF}, edges::Edges) where {N_MAX, TI, TF}
    edgesOnCell = cells.edges
    weights =ImmutableVectorArray(Vector{NTuple{N_MAX, Vec2Dxy{TF}}}(undef, cells.n), edgesOnCell.length)
    compute_weights_lsq2_velocity_reconstruction_periodic!(weights, cells.position, edges.position, edges.normal, edgesOnCell, cells.x_period, cells.y_period)
    return CellVelocityReconstructionLSq2(edges.n, edgesOnCell, weights)
end

function compute_weights_lsq2_velocity_reconstruction_spherical!(w::AbstractVector{<:ImmutableVector{N_MAX, Vec3D{TF}}}, ne, epos, cpos, edgesOnCell::AbstractVector{<:ImmutableVector{N_MAX}}, R::Number) where {TF, N_MAX}

    wdata = w.data
    @parallel for c in eachindex(edgesOnCell)
        @inbounds begin

        eoc = edgesOnCell[c]
        l = length(eoc)

        cp = cpos[c] / R

        xdir = zero(Vec3D{TF})
        ydir = zero(Vec3D{TF})

        M = Matrix{TF}(undef, l, 6)
        for i in Base.OneTo(l)
            e = eoc[i]
            _n = ne[e]
            n = normalize(_n - (_n ⋅ cp)*cp)

            if i == 1
                xdir = n
                ydir = normalize(cp × n)
                nx = oneunit(TF)
                ny = zero(TF)
            else
                nx = n ⋅ xdir
                ny = n ⋅ ydir
            end

            M[i, 1] = nx
            M[i, 2] = ny

            ep = epos[e] / R
            epp =ep * inv(ep ⋅ cp) 
            dp = epp - cp

            dx = dp ⋅ xdir
            dy = dp ⋅ ydir

            M[i, 3] = nx*dx
            M[i, 4] = ny*dx
            M[i, 5] = nx*dy
            M[i, 6] = ny*dy

        end

        MpM = Hermitian(M'*M)
        cn = cond(MpM)
        ii = 0
        while cn > 5e6
            ii += 1
            #@show c, ii, cn
            for i in 3:6
                MpM[i, i] += 1e-6
            end
            cn = cond(MpM)
        end

        P = LinearAlgebra.inv!(cholesky!(MpM))*M'

        wa = map(x -> @inbounds(P[1,x])*xdir + @inbounds(P[2,x])*ydir, Base.OneTo(l))
        w = ImmutableVector{N_MAX}(wa)

        wdata[c] = padwith(w, zero(Vec3D{TF})).data
        end #inbounds
    end
    return w
end

function CellVelocityReconstructionLSq2(cells::Cells{true, N_MAX, TI, TF}, edges::Edges) where {N_MAX, TI, TF}
    edgesOnCell = cells.edges
    weights =ImmutableVectorArray(Vector{NTuple{N_MAX, Vec3D{TF}}}(undef, cells.n), edgesOnCell.length)
    compute_weights_lsq2_velocity_reconstruction_spherical!(weights, edges.normal, edges.position, cells.position, edgesOnCell, cells.sphere_radius)
    return CellVelocityReconstructionLSq2(edges.n, edgesOnCell, weights)
end

function CellVelocityReconstructionLSq2(mesh::AbstractVoronoiMesh)
    CellVelocityReconstructionLSq2(mesh.cells, mesh.edges)
end

