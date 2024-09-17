abstract type CellVelocityReconstruction{N_MAX, TF, TI} <: LinearVoronoiOperator end
name_input(::CellVelocityReconstruction) = "edge"
name_output(::CellVelocityReconstruction) = "cell"

struct CellVelocityReconstructionPerot{N_MAX, TF, TI} <: CellVelocityReconstruction{N_MAX, TF, TI}
    n::Int
    indices::Vector{ImmutableVector{N_MAX, TI}}
    weights::Vector{ImmutableVector{N_MAX, Vec2Dxy{TF}}}
end

function compute_weights_perot_velocity_reconstruction_periodic!(w::Vector{ImmutableVector{N_MAX, T}}, c_pos, aC, Le, ne, edgesOnCell::Vector{<:ImmutableVector{N_MAX}}, v_pos, verticesOnEdge, xp::Number, yp::Number) where {T, N_MAX}

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
            r_vec = copysign(1, (ne[e] ⋅ r_vec)) * r_vec
            aux = @inbounds push(aux, inv_a * r_vec * Le[e])
        end
        w[c] = aux
        end #inbounds
    end
    return w
end

function CellVelocityReconstructionPerot(cells::CellInfo{false, N_MAX}, edges::EdgeInfo, vertices::VertexInfo, x_period::Number, y_period::Number) where {N_MAX}
    edgesOnCell = cells.indices.edges
    weights = Vector{ImmutableVector{N_MAX, eltype(cells.position)}}(undef, cells.n)
    compute_weights_perot_velocity_reconstruction_periodic!(weights, cells.position, cells.area, edges.dv, edges.normalVectors, edgesOnCell, vertices.position, edges.indices.vertices, x_period, y_period)
    return CellVelocityReconstructionPerot(edges.n, edgesOnCell, weights)
end

function CellVelocityReconstructionPerot(mesh::VoronoiMesh)
    isdefined(mesh.edges, :normalVectors) || compute_edge_normals!(mesh)
    CellVelocityReconstructionPerot(mesh.cells, mesh.edges, mesh.vertices, mesh.attributes[:x_period]::Float64, mesh.attributes[:y_period]::Float64)
end
