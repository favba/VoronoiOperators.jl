abstract type CellVelocityReconstruction{N_MAX, TI, TF, TZ} <: LinearVoronoiOperator end
name_input(::CellVelocityReconstruction) = "edge"
name_output(::CellVelocityReconstruction) = "cell"

struct CellVelocityReconstructionPerot{N_MAX, TI, TF, TZ} <: CellVelocityReconstruction{N_MAX, TI, TF, TZ}
    n::Int
    indices::SmVecArray{N_MAX, TI, 1}
    weights::SmVecArray{N_MAX, Vec{Union{TF, TZ}, 1, TF, TF, TZ}, 1}
end

method_name(::Type{<:CellVelocityReconstructionPerot}) = "Perot"

function perot_velocity_reconstruction_from_normal(vertices::SmallVector{N_MAX}, signEdge, base_point::Vec) where {N_MAX}

    inv_a = inv(area(vertices))

    @inbounds w = SmallVector{N_MAX, typeof(base_point)}()

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

function compute_weights_perot_velocity_reconstruction_periodic!(w::AbstractVector{SmallVector{N_MAX, T}}, c_pos,  v_pos, signEdge, verticesOnCell, xp::Number, yp::Number) where {T, N_MAX}

    wdata = w.data
    @parallel for c in eachindex(verticesOnCell)
        @inbounds begin
        cp = c_pos[c]
        voc = verticesOnCell[c]
        vertices = map(i -> closest(cp, @inbounds(v_pos[i]), xp, yp), voc)
        aux = perot_velocity_reconstruction_from_normal(vertices, signEdge[c], cp)
        wdata[c] = fixedvector(aux)
        end #inbounds
    end
    return w
end

function CellVelocityReconstructionPerot(cells::Cells{false, N_MAX, TI, TF}, edges::Edges, vertices::Vertices, x_period::Number, y_period::Number) where {N_MAX, TI, TF}
    edgesOnCell = cells.edges
    weights =SmallVectorArray(Vector{FixedVector{N_MAX, Vec2Dxy{TF}}}(undef, cells.n), edgesOnCell.length)
    compute_weights_perot_velocity_reconstruction_periodic!(weights, cells.position, vertices.position, cells.edgesSign, cells.vertices, x_period, y_period)
    return CellVelocityReconstructionPerot(edges.n, edgesOnCell, weights)
end

function CellVelocityReconstructionPerot(mesh::AbstractVoronoiMesh{false})
    CellVelocityReconstructionPerot(mesh.cells, mesh.edges, mesh.vertices, mesh.x_period, mesh.y_period)
end

function compute_weights_perot_velocity_reconstruction_spherical!(w::AbstractVector{SmallVector{N_MAX, T}}, R::Number, c_pos, v_pos, signEdges, verticesOnCell) where {T, N_MAX}

    wdata = w.data
    @parallel for c in eachindex(verticesOnCell)
        @inbounds begin
        cp_n = c_pos[c] / R

        voc = verticesOnCell[c]
        vertices = map(i -> (@inbounds(v_pos[i]) / R), voc)
        vertices_proj = project_points_to_tangent_plane(1, cp_n, vertices)
        aux = perot_velocity_reconstruction_from_normal(vertices_proj, signEdges[c], cp_n)
        wdata[c] = fixedvector(aux)
        end #inbounds
    end
    return w
end

function CellVelocityReconstructionPerot(cells::Cells{true, N_MAX, TI, TF}, edges::Edges{true}, vertices::Vertices{true}) where {N_MAX, TI, TF}
    edgesOnCell = cells.edges
    weights =SmallVectorArray(Vector{FixedVector{N_MAX, Vec3D{TF}}}(undef, cells.n), edgesOnCell.length)
    compute_weights_perot_velocity_reconstruction_spherical!(weights, cells.sphere_radius, cells.position, vertices.position, cells.edgesSign, cells.vertices)
    return CellVelocityReconstructionPerot(edges.n, edgesOnCell, weights)
end

function CellVelocityReconstructionPerot(mesh::AbstractVoronoiMesh{true})
    CellVelocityReconstructionPerot(mesh.cells, mesh.edges, mesh.vertices)
end

struct CellVelocityReconstructionLSq1{N_MAX, TI, TF, TZ} <: CellVelocityReconstruction{N_MAX, TI, TF, TZ}
    n::Int
    indices::SmVecArray{N_MAX, TI, 1}
    weights::SmVecArray{N_MAX, Vec{Union{TF, TZ}, 1, TF, TF, TZ}, 1}
end

method_name(::Type{<:CellVelocityReconstructionLSq1}) = "LSq1"

function CellVelocityReconstructionLSq1(cells::Cells{false, N_MAX, TI, TF}, edges::Edges) where {N_MAX, TI, TF}
    edgesOnCell = cells.edges
    weights =SmallVectorArray(Vector{FixedVector{N_MAX, Vec2Dxy{TF}}}(undef, cells.n), edgesOnCell.length)
    compute_weights_vec_lsq1_periodic!(weights, edgesOnCell, edges.normal)
    return CellVelocityReconstructionLSq1(edges.n, edgesOnCell, weights)
end

function CellVelocityReconstructionLSq1(cells::Cells{true, N_MAX, TI, TF}, edges::Edges) where {N_MAX, TI, TF}
    edgesOnCell = cells.edges
    weights =SmallVectorArray(Vector{FixedVector{N_MAX, Vec3D{TF}}}(undef, cells.n), edgesOnCell.length)
    compute_weights_vec_lsq1_spherical!(weights, cells.position, edgesOnCell, edges.normal, cells.sphere_radius)
    return CellVelocityReconstructionLSq1(edges.n, edgesOnCell, weights)
end

function CellVelocityReconstructionLSq1(mesh::AbstractVoronoiMesh)
    CellVelocityReconstructionLSq1(mesh.cells, mesh.edges)
end

struct CellVelocityReconstructionLSq2{N_MAX, TI, TF, TZ} <: CellVelocityReconstruction{N_MAX, TI, TF, TZ}
    n::Int
    indices::SmVecArray{N_MAX, TI, 1}
    weights::SmVecArray{N_MAX, Vec{Union{TF, TZ}, 1, TF, TF, TZ}, 1}
end

method_name(::Type{<:CellVelocityReconstructionLSq2}) = "LSq2"

function CellVelocityReconstructionLSq2(cells::Cells{false, N_MAX, TI, TF}, edges::Edges) where {N_MAX, TI, TF}
    edgesOnCell = cells.edges
    weights =SmallVectorArray(Vector{FixedVector{N_MAX, Vec2Dxy{TF}}}(undef, cells.n), edgesOnCell.length)
    compute_weights_vec_lsq2_periodic!(weights, cells.position, edgesOnCell, edges.position, edges.normal, cells.x_period, cells.y_period)
    return CellVelocityReconstructionLSq2(edges.n, edgesOnCell, weights)
end

function CellVelocityReconstructionLSq2(cells::Cells{true, N_MAX, TI, TF}, edges::Edges) where {N_MAX, TI, TF}
    edgesOnCell = cells.edges
    weights =SmallVectorArray(Vector{FixedVector{N_MAX, Vec3D{TF}}}(undef, cells.n), edgesOnCell.length)
    compute_weights_vec_lsq2_spherical!(weights, cells.position, edgesOnCell, edges.position, edges.normal, cells.sphere_radius)
    return CellVelocityReconstructionLSq2(edges.n, edgesOnCell, weights)
end

function CellVelocityReconstructionLSq2(mesh::AbstractVoronoiMesh)
    CellVelocityReconstructionLSq2(mesh.cells, mesh.edges)
end

#Different from CellVelocityReconstructionPerot only on the sphere
struct CellVelocityReconstructionPerotOld{N_MAX, TI, TF, TZ} <: CellVelocityReconstruction{N_MAX, TI, TF, TZ}
    n::Int
    indices::SmVecArray{N_MAX, TI, 1}
    weights::SmVecArray{N_MAX, Vec{Union{TF, TZ}, 1, TF, TF, TZ}, 1}
end

method_name(::Type{<:CellVelocityReconstructionPerotOld}) = "PerotOld"

function CellVelocityReconstructionPerotOld(cells::Cells{false, N_MAX, TI, TF}, edges::Edges, vertices::Vertices, x_period::Number, y_period::Number) where {N_MAX, TI, TF}
    edgesOnCell = cells.edges
    weights =SmallVectorArray(Vector{FixedVector{N_MAX, Vec2Dxy{TF}}}(undef, cells.n), edgesOnCell.length)
    compute_weights_perot_velocity_reconstruction_periodic!(weights, cells.position, vertices.position, cells.edgesSign, cells.vertices, x_period, y_period)
    return CellVelocityReconstructionPerotOld(edges.n, edgesOnCell, weights)
end

function CellVelocityReconstructionPerotOld(mesh::AbstractVoronoiMesh{false})
    CellVelocityReconstructionPerotOld(mesh.cells, mesh.edges, mesh.vertices, mesh.x_period, mesh.y_period)
end

function compute_weights_perotOld_velocity_reconstruction_spherical!(w::AbstractVector{SmallVector{N_MAX, T}}, R::Number, c_pos, areaCell, signEdges, edgesOnCell, e_mid_pos, edge_length) where {T, N_MAX}

    wdata = w.data
    @parallel for c in eachindex(edgesOnCell)
        @inbounds begin
        inv_a = inv(areaCell[c])
        cp = c_pos[c]
        cp_n = cp / R

        eoc = edgesOnCell[c]

        w = SmallVector{N_MAX, typeof(cp)}()
        signEdge_c = signEdges[c]
        l = length(eoc)
        @inbounds for ei in Base.OneTo(l)
            e = eoc[ei]
            mid_point = e_mid_pos[e]
            le = edge_length[e]
            ve_c = mid_point - cp
            ve_c_proj = ve_c - (ve_c ⋅ cp_n)*cp_n
            w = push(w, (signEdge_c[ei] * inv_a * le) * ve_c_proj)
        end

        wdata[c] = fixedvector(w)
        end #inbounds
    end
    return w
end

function CellVelocityReconstructionPerotOld(cells::Cells{true, N_MAX, TI, TF}, edges::Edges{true}, vertices::Vertices{true}) where {N_MAX, TI, TF}
    edgesOnCell = cells.edges
    weights =SmallVectorArray(Vector{FixedVector{N_MAX, Vec3D{TF}}}(undef, cells.n), edgesOnCell.length)
    compute_weights_perotOld_velocity_reconstruction_spherical!(weights, cells.sphere_radius, cells.position, cells.area, cells.edgesSign, cells.edges, edges.midpoint, edges.length)
    return CellVelocityReconstructionPerotOld(edges.n, edgesOnCell, weights)
end

function CellVelocityReconstructionPerotOld(mesh::AbstractVoronoiMesh{true})
    CellVelocityReconstructionPerotOld(mesh.cells, mesh.edges, mesh.vertices)
end

function save_cell_reconstruction(mesh::AbstractVoronoiMesh, method::String, output::String)
    if method == "Perot"
        VoronoiMeshes.save(output, CellVelocityReconstructionPerot(mesh))
    elseif method == "PerotOld"
        VoronoiMeshes.save(output, CellVelocityReconstructionPerotOld(mesh))
    elseif method == "LSq1"
        VoronoiMeshes.save(output, CellVelocityReconstructionLSq1(mesh))
    elseif method == "LSq2"
        VoronoiMeshes.save(output, CellVelocityReconstructionLSq2(mesh))
    else
        throw(error(string(method, " is not a valid cell velocity reconstruction method")))
    end
end

