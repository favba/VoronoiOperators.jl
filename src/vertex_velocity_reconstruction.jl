abstract type VertexVelocityReconstruction{TI, TF, TZ} <: LinearVoronoiOperator end
name_input(::VertexVelocityReconstruction) = "edge"
name_output(::VertexVelocityReconstruction) = "vertex"

struct VertexVelocityReconstructionPerot{TI, TF, TZ} <: VertexVelocityReconstruction{TI, TF, TZ}
    n::Int
    indices::Vector{FixedVector{3,TI}}
    weights::Vector{FixedVector{3, TensorsLite.VecMaybe2Dxy{TF,TZ}}}
end

method_name(::Type{<:VertexVelocityReconstructionPerot}) = "Perot"

function perot_velocity_reconstruction_from_tangent(vertices::SmallVector{N_MAX}, signEdge, base_point::Vec, base_normal::Vec) where {N_MAX}

    inv_a = inv(area(vertices))

    @inbounds w = SmallVector{N_MAX, typeof(base_point)}()

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

function compute_weights_perot_vertex_velocity_reconstruction_periodic!(w::AbstractVector{FixedVector{3, Vec2Dxy{TF}}}, vertexPos, vertexEdgeSign, cellsOnVertex::AbstractVector{<:FixedVector{3}}, cellPos, xp::Number, yp::Number) where {TF}

    @parallel for v in eachindex(cellsOnVertex)
        @inbounds begin
        vp = vertexPos[v]

        cov = cellsOnVertex[v]

        points = SmallVector{3,Vec2Dxy{TF}}(map(i -> closest(vp, @inbounds(cellPos[i]), xp , yp), cov), UInt(3))
        aux = perot_velocity_reconstruction_from_tangent(points, vertexEdgeSign[v], vp, 𝐤)

        w[v] = fixedvector(aux)
        end #inbounds
    end
    return w
end

function compute_weights_perot_vertex_velocity_reconstruction(cells::Cells{false, NE, TI, TF}, vertices::Vertices{false, NE, TI, TF}) where {NE, TI, TF}
    w = Vector{FixedVector{3, Vec2Dxy{TF}}}(undef, vertices.n)
    return compute_weights_perot_vertex_velocity_reconstruction_periodic!(w, vertices.position, vertices.edgesSign, vertices.cells, cells.position, cells.x_period, cells.y_period)
end

function compute_weights_perot_vertex_velocity_reconstruction_sphere!(w::AbstractVector{FixedVector{3, Vec3D{TF}}}, R::Number, vertexPos, vertexEdgeSign, cellsOnVertex::AbstractVector{<:FixedVector{3}}, cellPos) where {TF}

    @parallel for v in eachindex(cellsOnVertex)
        @inbounds begin
        vp = vertexPos[v] / R

        cov = cellsOnVertex[v]

        points = SmallVector{3,Vec3D{TF}}(map(i -> (@inbounds(cellPos[i]) / R), cov), UInt(3))
        points_projected = project_points_to_tangent_plane(1, vp, points)
        aux = perot_velocity_reconstruction_from_tangent(points_projected, vertexEdgeSign[v], vp, vp)

        w[v] = fixedvector(aux)
        end #inbounds
    end
    return w
end

function compute_weights_perot_vertex_velocity_reconstruction(cells::Cells{true, NE, TI, TF}, vertices::Vertices{true, NE, TI, TF}) where {NE, TI, TF}
    w = Vector{FixedVector{3, Vec3D{TF}}}(undef, vertices.n)
    return compute_weights_perot_vertex_velocity_reconstruction_sphere!(w, cells.sphere_radius, vertices.position, vertices.edgesSign, vertices.cells, cells.position)
end

VertexVelocityReconstructionPerot(mesh::AbstractVoronoiMesh) =
    VertexVelocityReconstructionPerot(
        mesh.edges.n,
        mesh.vertices.edges,
        compute_weights_perot_vertex_velocity_reconstruction(mesh.cells, mesh.vertices)
    )

struct VertexVelocityReconstructionLSq1{TI, TF, TZ} <: VertexVelocityReconstruction{TI, TF, TZ}
    n::Int
    indices::Vector{FixedVector{3, TI}}
    weights::Vector{FixedVector{3, Tensor{Union{TF, TZ}, 1, TF, TF, TZ}}}
end

method_name(::Type{<:VertexVelocityReconstructionLSq1}) = "LSq1"

function VertexVelocityReconstructionLSq1(vertices::Vertices{false, NE, TI, TF}, edges::Edges) where {NE, TI, TF}
    edgesOnVertex = vertices.edges
    weights = Vector{FixedVector{3, Vec2Dxy{TF}}}(undef, vertices.n)
    compute_weights_vec_lsq1_periodic!(weights, edgesOnVertex, edges.normal)
    return VertexVelocityReconstructionLSq1(edges.n, edgesOnVertex, weights)
end

function VertexVelocityReconstructionLSq1(vertices::Vertices{true, NE, TI, TF}, edges::Edges) where {NE, TI, TF}
    edgesOnVertex = vertices.edges
    weights =Vector{FixedVector{3, Vec3D{TF}}}(undef, vertices.n)
    compute_weights_vec_lsq1_spherical!(weights, vertices.position, edgesOnVertex, edges.normal, vertices.sphere_radius)
    return VertexVelocityReconstructionLSq1(edges.n, edgesOnVertex, weights)
end

function VertexVelocityReconstructionLSq1(mesh::AbstractVoronoiMesh)
    VertexVelocityReconstructionLSq1(mesh.vertices, mesh.edges)
end

struct VertexVelocityReconstructionLSq2{TI, TF, TZ} <: VertexVelocityReconstruction{TI, TF, TZ}
    n::Int
    indices::Vector{FixedVector{3, TI}}
    weights::Vector{FixedVector{3, Tensor{Union{TF, TZ}, 1, TF, TF, TZ}}}
end

method_name(::Type{<:VertexVelocityReconstructionLSq2}) = "LSq2"

function VertexVelocityReconstructionLSq2(vertices::Vertices{false, NE, TI, TF}, edges::Edges) where {NE, TI, TF}
    edgesOnVertex = vertices.edges
    weights = Vector{FixedVector{3, Vec2Dxy{TF}}}(undef, vertices.n)
    compute_weights_vec_lsq2_periodic!(weights, vertices.position, edgesOnVertex, edges.position, edges.normal, vertices.x_period, vertices.y_period)
    return VertexVelocityReconstructionLSq2(edges.n, edgesOnVertex, weights)
end

function VertexVelocityReconstructionLSq2(vertices::Vertices{true, NE, TI, TF}, edges::Edges) where {NE, TI, TF}
    edgesOnVertex = vertices.edges
    weights = Vector{FixedVector{3, Vec3D{TF}}}(undef, vertices.n)
    compute_weights_vec_lsq2_spherical!(weights, vertices.position, edgesOnVertex, edges.position, edges.normal, vertices.sphere_radius)
    return VertexVelocityReconstructionLSq2(edges.n, edgesOnVertex, weights)
end

function VertexVelocityReconstructionLSq2(mesh::AbstractVoronoiMesh)
    VertexVelocityReconstructionLSq2(mesh.vertices, mesh.edges)
end

function save_vertex_reconstruction(mesh::AbstractVoronoiMesh, method::String, output::String)
    if method == "Perot"
        VoronoiMeshes.save(output, VertexVelocityReconstructionPerot(mesh))
    elseif method == "LSq1"
        VoronoiMeshes.save(output, VertexVelocityReconstructionLSq1(mesh))
    elseif method == "LSq2"
        VoronoiMeshes.save(output, VertexVelocityReconstructionLSq2(mesh))
    else
        throw(error(string(method, " is not a valid vertex velocity reconstruction method")))
    end
end

