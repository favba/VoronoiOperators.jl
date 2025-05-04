abstract type VertexVelocityReconstruction{TI, TF, TZ} <: LinearVoronoiOperator end
name_input(::VertexVelocityReconstruction) = "edge"
name_output(::VertexVelocityReconstruction) = "vertex"

struct VertexVelocityReconstructionPerot{TI, TF, TZ} <: VertexVelocityReconstruction{TI, TF, TZ}
    n::Int
    indices::Vector{NTuple{3,TI}}
    weights::Vector{NTuple{3, TensorsLite.VecMaybe2Dxy{TF,TZ}}}
end

function perot_velocity_reconstruction_from_tangent(vertices::ImmutableVector{N_MAX}, signEdge, base_point::Vec, base_normal::Vec) where {N_MAX}

    inv_a = inv(area(vertices))

    @inbounds w = ImmutableVector{N_MAX, typeof(base_point)}()

    l = length(vertices)
    vert1 = @inbounds(vertices[l])
    @inbounds for vi in Base.OneTo(l)
        vert2 = vertices[vi]
        mid_point = (vert2 + vert1) / 2
        le = norm(vert2 - vert1)
        w = push(w, base_normal × (signEdge[vi] * inv_a * le * (mid_point - base_point)))
        vert1 = vert2
    end

    return w
end

function compute_weights_perot_vertex_velocity_reconstruction_periodic!(w::AbstractVector{NTuple{3, Vec2Dxy{TF}}}, vertexPos, vertexEdgeSign, cellsOnVertex::AbstractVector{<:NTuple{3}}, cellPos, xp::Number, yp::Number) where {TF}

    @parallel for v in eachindex(cellsOnVertex)
        @inbounds begin
        vp = vertexPos[v]

        cov = cellsOnVertex[v]

        points = ImmutableVector{3,Vec2Dxy{TF}}(map(i -> closest(vp, @inbounds(cellPos[i]), xp , yp), cov), UInt(3))
        aux = perot_velocity_reconstruction_from_tangent(points, vertexEdgeSign[v], vp, 𝐤)

        w[v] = aux.data
        end #inbounds
    end
    return w
end

function compute_weights_perot_vertex_velocity_reconstruction(cells::Cells{false, NE, TI, TF}, vertices::Vertices{false, NE, TI, TF}) where {NE, TI, TF}
    w = Vector{NTuple{3, Vec2Dxy{TF}}}(undef, vertices.n)
    return compute_weights_perot_vertex_velocity_reconstruction_periodic!(w, vertices.position, vertices.edgesSign, vertices.cells, cells.position, cells.x_period, cells.y_period)
end

function compute_weights_perot_vertex_velocity_reconstruction_sphere!(w::AbstractVector{NTuple{3, Vec3D{TF}}}, R::Number, vertexPos, vertexEdgeSign, cellsOnVertex::AbstractVector{<:NTuple{3}}, cellPos) where {TF}

    @parallel for v in eachindex(cellsOnVertex)
        @inbounds begin
        vp = vertexPos[v] / R

        cov = cellsOnVertex[v]

        points = ImmutableVector{3,Vec3D{TF}}(map(i -> (@inbounds(cellPos[i]) / R), cov), UInt(3))
        points_projected = project_points_to_tangent_plane(1, vp, points)
        aux = perot_velocity_reconstruction_from_tangent(points_projected, vertexEdgeSign[v], vp, vp)

        w[v] = aux.data
        end #inbounds
    end
    return w
end

function compute_weights_perot_vertex_velocity_reconstruction(cells::Cells{true, NE, TI, TF}, vertices::Vertices{true, NE, TI, TF}) where {NE, TI, TF}
    w = Vector{NTuple{3, Vec3D{TF}}}(undef, vertices.n)
    return compute_weights_perot_vertex_velocity_reconstruction_sphere!(w, cells.sphere_radius, vertices.position, vertices.edgesSign, vertices.cells, cells.position)
end

VertexVelocityReconstructionPerot(mesh::AbstractVoronoiMesh) =
    VertexVelocityReconstructionPerot(
        mesh.edges.n,
        mesh.vertices.edges,
        compute_weights_perot_vertex_velocity_reconstruction(mesh.cells, mesh.vertices)
    )

