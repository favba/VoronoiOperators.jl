abstract type CellVelocityReconstruction{N_MAX, TI, TF} <: LinearVoronoiOperator end
name_input(::CellVelocityReconstruction) = "edge"
name_output(::CellVelocityReconstruction) = "cell"

struct CellVelocityReconstructionPerot{N_MAX, TI, TF, TZ} <: CellVelocityReconstruction{N_MAX, TI, TF}
    n::Int
    indices::ImVecArray{N_MAX, TI, 1}
    weights::Vector{ImmutableVector{N_MAX, Vec{Union{TF, TZ}, 1, TF, TF, TZ}}}
end

function compute_weights_perot_velocity_reconstruction_periodic!(w::AbstractVector{ImmutableVector{N_MAX, T}}, c_pos, aC, Le, ne, edgesOnCell::AbstractVector{<:ImmutableVector{N_MAX}}, v_pos, verticesOnEdge, xp::Number, yp::Number) where {T, N_MAX}

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
        w[c] = aux
        end #inbounds
    end
    return w
end

function CellVelocityReconstructionPerot(cells::Cells{false, N_MAX}, edges::Edges, vertices::Vertices, x_period::Number, y_period::Number) where {N_MAX}
    edgesOnCell = cells.edges
    weights = Vector{ImmutableVector{N_MAX, eltype(cells.position)}}(undef, cells.n)
    compute_weights_perot_velocity_reconstruction_periodic!(weights, cells.position, cells.area, edges.length, edges.normal, edgesOnCell, vertices.position, edges.vertices, x_period, y_period)
    return CellVelocityReconstructionPerot(edges.n, edgesOnCell, weights)
end

function CellVelocityReconstructionPerot(mesh::VoronoiMesh{false})
    CellVelocityReconstructionPerot(mesh.cells, mesh.edges, mesh.vertices, mesh.x_period, mesh.y_period)
end

function compute_weights_perot_velocity_reconstruction_spherical!(w::AbstractVector{ImmutableVector{N_MAX, T}}, R::Number, c_pos, aC, Le, ne, edgesOnCell::AbstractVector{<:ImmutableVector{N_MAX}}, v_pos, verticesOnEdge) where {T, N_MAX}

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
        w[c] = aux
        end #inbounds
    end
    return w
end

function CellVelocityReconstructionPerot(cells::Cells{true, N_MAX}, edges::Edges{true}, vertices::Vertices{true}) where {N_MAX}
    edgesOnCell = cells.edges
    weights = Vector{ImmutableVector{N_MAX, eltype(cells.position)}}(undef, cells.n)
    compute_weights_perot_velocity_reconstruction_spherical!(weights, cells.sphere_radius, cells.position, cells.area, edges.length, edges.normal, edgesOnCell, vertices.position, edges.vertices)
    return CellVelocityReconstructionPerot(edges.n, edgesOnCell, weights)
end

function CellVelocityReconstructionPerot(mesh::VoronoiMesh{true})
    CellVelocityReconstructionPerot(mesh.cells, mesh.edges, mesh.vertices)
end
