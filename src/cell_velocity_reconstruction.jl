abstract type CellVelocityReconstruction{N_MAX, TI, TF, TZ} <: LinearVoronoiOperator end
name_input(::CellVelocityReconstruction) = "edge"
name_output(::CellVelocityReconstruction) = "cell"

struct CellVelocityReconstructionPerot{N_MAX, TI, TF, TZ} <: CellVelocityReconstruction{N_MAX, TI, TF, TZ}
    n::Int
    indices::ImVecArray{N_MAX, TI, 1}
    weights::ImVecArray{N_MAX, Vec{Union{TF, TZ}, 1, TF, TF, TZ}, 1}
end

function perot_velocity_reconstruction_from_normal(vertices::ImmutableVector{N_MAX}, signEdge, base_point::Vec) where {N_MAX}

    inv_a = inv(area(vertices))

    @inbounds w = ImmutableVector{N_MAX, typeof(base_point)}()

    l = length(vertices)
    vert1 = @inbounds(vertices[l])
    @inbounds for vi in Base.OneTo(l)
        vert2 = vertices[vi]
        mid_point = (vert2 + vert1) / 2
        le = norm(vert2 - vert1)
        w = push(w, signEdge[vi] * inv_a * le * (mid_point - base_point))
        vert1 = vert2
    end

    return w
end

function compute_weights_perot_velocity_reconstruction_periodic!(w::AbstractVector{ImmutableVector{N_MAX, T}}, c_pos,  v_pos, signEdge, verticesOnCell, xp::Number, yp::Number) where {T, N_MAX}

    wdata = w.data
    @parallel for c in eachindex(verticesOnCell)
        @inbounds begin
        cp = c_pos[c]
        voc = verticesOnCell[c]
        vertices = map(i -> closest(cp, @inbounds(v_pos[i]), xp, yp), voc)
        aux = perot_velocity_reconstruction_from_normal(vertices, signEdge[c], cp)
        wdata[c] = padwith(aux, zero(T)).data
        end #inbounds
    end
    return w
end

function CellVelocityReconstructionPerot(cells::Cells{false, N_MAX, TI, TF}, edges::Edges, vertices::Vertices, x_period::Number, y_period::Number) where {N_MAX, TI, TF}
    edgesOnCell = cells.edges
    weights =ImmutableVectorArray(Vector{NTuple{N_MAX, Vec2Dxy{TF}}}(undef, cells.n), edgesOnCell.length)
    compute_weights_perot_velocity_reconstruction_periodic!(weights, cells.position, vertices.position, cells.edgesSign, cells.vertices, x_period, y_period)
    return CellVelocityReconstructionPerot(edges.n, edgesOnCell, weights)
end

function CellVelocityReconstructionPerot(mesh::AbstractVoronoiMesh{false})
    CellVelocityReconstructionPerot(mesh.cells, mesh.edges, mesh.vertices, mesh.x_period, mesh.y_period)
end

function compute_weights_perot_velocity_reconstruction_spherical!(w::AbstractVector{ImmutableVector{N_MAX, T}}, R::Number, c_pos, v_pos, signEdges, verticesOnCell) where {T, N_MAX}

    wdata = w.data
    @parallel for c in eachindex(verticesOnCell)
        @inbounds begin
        cp_n = c_pos[c] / R

        voc = verticesOnCell[c]
        vertices = map(i -> (@inbounds(v_pos[i]) / R), voc)
        vertices_proj = project_points_to_tangent_plane(1, cp_n, vertices)
        aux = perot_velocity_reconstruction_from_normal(vertices_proj, signEdges[c], cp_n)
        wdata[c] = padwith(aux, zero(T)).data
        end #inbounds
    end
    return w
end

function CellVelocityReconstructionPerot(cells::Cells{true, N_MAX, TI, TF}, edges::Edges{true}, vertices::Vertices{true}) where {N_MAX, TI, TF}
    edgesOnCell = cells.edges
    weights =ImmutableVectorArray(Vector{NTuple{N_MAX, Vec3D{TF}}}(undef, cells.n), edgesOnCell.length)
    compute_weights_perot_velocity_reconstruction_spherical!(weights, cells.sphere_radius, cells.position, vertices.position, cells.edgesSign, cells.vertices)
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

