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

struct CellVelocityReconstructionLSQ1{N_MAX, TI, TF, TZ} <: CellVelocityReconstruction{N_MAX, TI, TF, TZ}
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

        wx = vec(P[1,:])
        wy = vec(P[2,:])

        wa = map(x -> Vec(x = x[1], y = x[2]), zip(wx, wy))
        w = ImmutableVector{N_MAX}(wa)

        wdata[c] = padwith(w, zero(Vec2Dxy{TF})).data
        end #inbounds
    end
    return w
end

function CellVelocityReconstructionLSQ1(cells::Cells{false, N_MAX, TI, TF}, edges::Edges) where {N_MAX, TI, TF}
    edgesOnCell = cells.edges
    weights =ImmutableVectorArray(Vector{NTuple{N_MAX, Vec2Dxy{TF}}}(undef, cells.n), edgesOnCell.length)
    compute_weights_interp_velocity_reconstruction_periodic!(weights, edges.normal, edgesOnCell)
    return CellVelocityReconstructionLSQ1(edges.n, edgesOnCell, weights)
end

function compute_weights_interp_velocity_reconstruction_spherical!(w::AbstractVector{<:ImmutableVector{N_MAX, Vec3D{TF}}}, ne, cpos, edgesOnCell::AbstractVector{<:ImmutableVector{N_MAX}}, R::Number) where {TF, N_MAX}

    wdata = w.data
    @parallel for c in eachindex(edgesOnCell)
        @inbounds begin

        eoc = edgesOnCell[c]
        l = length(eoc)

        cn = cpos[c] / R

        xdir = zero(Vec3D{TF})
        ydir = zero(Vec3D{TF})

        M = Matrix{TF}(undef, l, 2)
        for i in Base.OneTo(l)
            e = eoc[i]
            _n = ne[e]
            n = normalize(_n - (_n ⋅ cn)*cn)
            if i == 1
                xdir = n
                ydir = normalize(cn × n)
                M[i, 1] = oneunit(TF)
                M[i, 2] = zero(TF)
            else
                M[i, 1] = n ⋅ xdir
                M[i, 2] = n ⋅ ydir
            end
        end
        MpM = cholesky!(Hermitian(M'*M))
        P = LinearAlgebra.inv!(MpM)*M'

        wx = vec(P[1,:])
        wy = vec(P[2,:])

        wa = map(x -> x[1]*xdir + x[2]*ydir, zip(wx, wy))
        w = ImmutableVector{N_MAX}(wa)

        wdata[c] = padwith(w, zero(Vec3D{TF})).data
        end #inbounds
    end
    return w
end

function CellVelocityReconstructionLSQ1(cells::Cells{true, N_MAX, TI, TF}, edges::Edges) where {N_MAX, TI, TF}
    edgesOnCell = cells.edges
    weights =ImmutableVectorArray(Vector{NTuple{N_MAX, Vec3D{TF}}}(undef, cells.n), edgesOnCell.length)
    compute_weights_interp_velocity_reconstruction_spherical!(weights, edges.normal, cells.position, edgesOnCell, cells.sphere_radius)
    return CellVelocityReconstructionLSQ1(edges.n, edgesOnCell, weights)
end

function CellVelocityReconstructionLSQ1(mesh::AbstractVoronoiMesh)
    CellVelocityReconstructionLSQ1(mesh.cells, mesh.edges)
end

