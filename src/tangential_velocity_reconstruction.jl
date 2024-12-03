abstract type TangentialVelocityReconstruction{N_MAX, TI, TF} <: LinearVoronoiOperator end
name_input(::TangentialVelocityReconstruction) = "edge"
name_output(::TangentialVelocityReconstruction) = "edge"
n_input(a::TangentialVelocityReconstruction) = n_output(a)

struct TangentialVelocityReconstructionThuburn{N_MAX, TI, TF} <: TangentialVelocityReconstruction{N_MAX, TI, TF}
    indices::ImVecArray{N_MAX, TI, 1}
    weights::ImVecArray{N_MAX, TF, 1}
end

function compute_weightsOnEdge_trisk!(edgesOnEdge::AbstractVector{<:ImmutableVector{NE, TI}}, weightsOnEdge::AbstractVector{<:ImmutableVector{NE, TF}}, cellsOnEdge, verticesOnCell, edgesOnCell, dcEdge, dvEdge, kiteAreasOnVertex, cellsOnVertex, nEdgesOnCell, areaCell) where {NE, TI, TF}

    @parallel for e in eachindex(cellsOnEdge)
        @inbounds begin

        w = ImmutableVector{NE, TF}()
        inds = ImmutableVector{NE, TI}()
        c1,c2 = cellsOnEdge[e]
        inv_de = inv(dcEdge[e])

        inds_e_c1 = edgesOnCell[c1]
        this_e_i = findfirst(isequal(e), inds_e_c1)
        inds_e_c1_shifted = circshift(inds_e_c1, -this_e_i)
        vertices_c1_shifted = circshift(verticesOnCell[c1], -this_e_i)
        nEdges_c1 = nEdgesOnCell[c1]
        inv_a_c1 = inv(areaCell[c1])

        R = zero(TF)
        for i in 1:(nEdges_c1-1)
            next_e = inds_e_c1_shifted[i]
            v = vertices_c1_shifted[i]
            Avi = VoronoiMeshes.select_kite_area(kiteAreasOnVertex,cellsOnVertex,v,c1)
            R += inv_a_c1*Avi
            w = push(w, VoronoiMeshes.sign_edge(cellsOnEdge[next_e],c1)*inv_de*dvEdge[next_e]*(0.5 - R))
            inds = push(inds, next_e)
        end

        inds_e_c2 = edgesOnCell[c2]
        this_e_i = findfirst(isequal(e), inds_e_c2)
        inds_e_c2_shifted = circshift(inds_e_c2, -this_e_i)
        vertices_c2_shifted = circshift(verticesOnCell[c2], -this_e_i)
        nEdges_c2 = nEdgesOnCell[c2]
        inv_a_c2 = inv(areaCell[c2])

        R = zero(TF)
        for i in 1:(nEdges_c2-1)
            next_e = inds_e_c2_shifted[i]
            v = vertices_c2_shifted[i]
            Avi = VoronoiMeshes.select_kite_area(kiteAreasOnVertex,cellsOnVertex,v,c2)
            R += inv_a_c2*Avi
            w = push(w, (-VoronoiMeshes.sign_edge(cellsOnEdge[next_e],c2))*inv_de*dvEdge[next_e]*(0.5 - R))
            inds = push(inds, next_e)
        end

        edgesOnEdge[e] = inds
        weightsOnEdge[e] = padwith(w, zero(TF))
        end #inbounds
    end
    return edgesOnEdge, weightsOnEdge
end

function compute_weightsOnEdge_trisk!(indices, w, cells::Cells,vertices::Vertices, edges::Edges)
    return compute_weightsOnEdge_trisk!(indices, w, edges.cells, cells.vertices, cells.edges, edges.lengthDual, edges.length, vertices.kiteAreas, vertices.cells, cells.nEdges, cells.area)
end

compute_weightsOnEdge_trisk!(indices, weightsOnEdge,mesh::AbstractVoronoiMesh) = compute_weightsOnEdge_trisk!(indices, weightsOnEdge, mesh.cells, mesh.vertices, mesh.edges)

function compute_weightsOnEdge_trisk(mesh::AbstractVoronoiMesh)
    nEdgesOnEdge = maximum(x->(x[1] + x[2] - 2), ((a,inds) -> @inbounds((a[inds[1]], a[inds[2]]))).((mesh.cells.nEdges,), mesh.edges.cells))
    indices =ImVecArray{nEdgesOnEdge, integer_type(mesh)}(mesh.edges.n)
    weights =ImmutableVectorArray(Vector{NTuple{nEdgesOnEdge, float_type(mesh)}}(undef, mesh.edges.n), indices.length)
    return compute_weightsOnEdge_trisk!(indices, weights, mesh)
end

function TangentialVelocityReconstructionThuburn(mesh::AbstractVoronoiMesh)
    return TangentialVelocityReconstructionThuburn(compute_weightsOnEdge_trisk(mesh)...)
end

struct TangentialVelocityReconstructionPeixoto{N_MAX, TI, TF} <: TangentialVelocityReconstruction{N_MAX, TI, TF}
    indices::ImVecArray{N_MAX, TI, 1}
    weights::ImVecArray{N_MAX, TF, 1}
end

function compute_weightsOnEdge_Peixoto_periodic!(
            edgesOnEdge::AbstractVector{<:ImmutableVector{NE, TI}},
            weightsOnEdge::AbstractVector{<:ImmutableVector{NE, TF}},
            edge_pos,
            cell_pos,
            v_pos,
            cellsOnEdge,
            verticesOnEdge,
            edgesOnCell,
            dvEdge,
            nEdgesOnCell,
            areaCell,
            xp::Number, yp::Number
    ) where {NE, TI, TF}

    @parallel for e in eachindex(cellsOnEdge)
        @inbounds begin

        w = ImmutableVector{NE, TF}()
        inds = ImmutableVector{NE, TI}()

        ep = edge_pos[e]
        c1,c2 = cellsOnEdge[e]
        c1e = closest(ep, cell_pos[c1], xp ,yp)
        c2e = closest(ep, cell_pos[c2], xp ,yp)

        v1,v2 = verticesOnEdge[e]
        v1e = closest(ep, v_pos[v1], xp ,yp)
        v2e = closest(ep, v_pos[v2], xp ,yp)
        te = normalize(v2e - v1e)

        inds_e_c1 = edgesOnCell[c1]
        this_e_i = findfirst(isequal(e), inds_e_c1)
        inds_e_c1_shifted = circshift(inds_e_c1, -this_e_i)
        nEdges_c1 = Int(nEdgesOnCell[c1])
        a_c1 = areaCell[c1]

        inds_e_c2 = edgesOnCell[c2]
        this_e_i = findfirst(isequal(e), inds_e_c2)
        inds_e_c2_shifted = circshift(inds_e_c2, -this_e_i)
        nEdges_c2 = Int(nEdgesOnCell[c2])
        a_c2 = areaCell[c2]

        for i in 1:(nEdges_c1-1)
            e_ = inds_e_c1_shifted[i]
            v1, v2 = verticesOnEdge[e_]
            v1p = closest(ep, v_pos[v1], xp, yp)
            v2p = closest(ep, v_pos[v2], xp, yp)
            emid = (v1p + v2p) / 2

            r_vec = c1e - emid

            c1, c2 = cellsOnEdge[e_]
            c1p = closest(ep, cell_pos[c1], xp, yp)
            c2p = closest(ep, cell_pos[c2], xp, yp)
            ne = normalize(c2p - c1p)

            r_vec = sign(ne ⋅ r_vec) * r_vec
            w = push(w,  (dvEdge[e_] / (2 * a_c1)) * (r_vec ⋅ te))
            inds = push(inds, e_)
        end

        for i in 1:(nEdges_c2-1)
            e_ = inds_e_c2_shifted[i]
            v1, v2 = verticesOnEdge[e_]
            v1p = closest(ep, v_pos[v1], xp, yp)
            v2p = closest(ep, v_pos[v2], xp, yp)
            emid = (v1p + v2p) / 2

            r_vec = c2e - emid

            c1, c2 = cellsOnEdge[e_]
            c1p = closest(ep, cell_pos[c1], xp, yp)
            c2p = closest(ep, cell_pos[c2], xp, yp)
            ne = normalize(c2p - c1p)

            r_vec = sign(ne ⋅ r_vec) * r_vec
            w = push(w,  (dvEdge[e_] / (2 * a_c2)) * (r_vec ⋅ te))
            inds = push(inds, e_)
        end

        edgesOnEdge[e] = inds
        weightsOnEdge[e] = padwith(w, zero(TF))
        end #inbounds
    end
    return edgesOnEdge, weightsOnEdge
end

function compute_weightsOnEdge_Peixoto!(indices, w, cells::Cells{false},vertices::Vertices{false}, edges::Edges{false})
    return compute_weightsOnEdge_Peixoto_periodic!(
        indices,
        w,
        edges.position,
        cells.position,
        vertices.position,
        edges.cells,
        edges.vertices,
        cells.edges,
        edges.length,
        cells.nEdges,
        cells.area,
        cells.x_period, cells.y_period
    )
end

function compute_weightsOnEdge_Peixoto_spherical!(
            edgesOnEdge::AbstractVector{<:ImmutableVector{NE, TI}},
            weightsOnEdge::AbstractVector{<:ImmutableVector{NE, TF}},
            R::Number,
            cell_pos,
            v_pos,
            edge_tan,
            cellsOnEdge,
            verticesOnEdge,
            edgesOnCell,
            dvEdge,
            nEdgesOnCell,
            areaCell
    ) where {NE, TI, TF}

    @parallel for e in eachindex(cellsOnEdge)
        @inbounds begin

        w = ImmutableVector{NE, TF}()
        inds = ImmutableVector{NE, TI}()

        c1,c2 = cellsOnEdge[e]
        c1e = cell_pos[c1]
        c1en = c1e / R
        c2e = cell_pos[c2]
        c2en = c2e / R

        te = edge_tan[e]
        te_c1 = normalize(te - (te ⋅ c1en)*c1en)
        te_c2 = normalize(te - (te ⋅ c2en)*c2en)

        inds_e_c1 = edgesOnCell[c1]
        this_e_i = findfirst(isequal(e), inds_e_c1)
        inds_e_c1_shifted = circshift(inds_e_c1, -this_e_i)
        nEdges_c1 = Int(nEdgesOnCell[c1])
        a_c1 = areaCell[c1]

        inds_e_c2 = edgesOnCell[c2]
        this_e_i = findfirst(isequal(e), inds_e_c2)
        inds_e_c2_shifted = circshift(inds_e_c2, -this_e_i)
        nEdges_c2 = Int(nEdgesOnCell[c2])
        a_c2 = areaCell[c2]

        for i in 1:(nEdges_c1-1)
            e_ = inds_e_c1_shifted[i]
            v1, v2 = verticesOnEdge[e_]
            v1p = v_pos[v1]
            v2p = v_pos[v2]
            emid_n = normalize((v1p + v2p) / 2)
            emid = emid_n * (R / (emid_n ⋅ c1en))

            r_vec = c1e - emid

            c1, c2 = cellsOnEdge[e_]
            c1p = cell_pos[c1]
            c2p = cell_pos[c2]
            ne = normalize(c2p - c1p)

            r_vec = sign(ne ⋅ r_vec) * r_vec
            w = push(w,  (dvEdge[e_] / (2 * a_c1)) * (r_vec ⋅ te_c1))
            inds = push(inds, e_)
        end

        for i in 1:(nEdges_c2-1)
            e_ = inds_e_c2_shifted[i]
            v1, v2 = verticesOnEdge[e_]
            v1p = v_pos[v1]
            v2p = v_pos[v2]
            emid_n = normalize((v1p + v2p) / 2)
            emid = emid_n * (R / (emid_n ⋅ c2en))

            r_vec = c2e - emid

            c1, c2 = cellsOnEdge[e_]
            c1p = cell_pos[c1]
            c2p = cell_pos[c2]
            ne = normalize(c2p - c1p)

            r_vec = sign(ne ⋅ r_vec) * r_vec
            w = push(w,  (dvEdge[e_] / (2 * a_c2)) * (r_vec ⋅ te_c2))
            inds = push(inds, e_)
        end

        edgesOnEdge[e] = inds
        weightsOnEdge[e] = padwith(w, zero(TF))
        end #inbounds
    end
    return edgesOnEdge, weightsOnEdge
end

function compute_weightsOnEdge_Peixoto!(indices, w, cells::Cells{true},vertices::Vertices{true}, edges::Edges{true})
    return compute_weightsOnEdge_Peixoto_spherical!(
        indices,
        w,
        cells.sphere_radius,
        cells.position,
        vertices.position,
        edges.tangent,
        edges.cells,
        edges.vertices,
        cells.edges,
        edges.length,
        cells.nEdges,
        cells.area,
    )
end

compute_weightsOnEdge_Peixoto!(indices, weightsOnEdge,mesh::AbstractVoronoiMesh) = compute_weightsOnEdge_Peixoto!(indices, weightsOnEdge, mesh.cells, mesh.vertices, mesh.edges)

function compute_weightsOnEdge_Peixoto(mesh::AbstractVoronoiMesh)
    nEdgesOnEdge = maximum(x->(x[1] + x[2] - 2), ((a,inds) -> @inbounds((a[inds[1]], a[inds[2]]))).((mesh.cells.nEdges,), mesh.edges.cells))
    indices =ImVecArray{nEdgesOnEdge, integer_type(mesh)}(mesh.edges.n)
    weights =ImmutableVectorArray(Vector{NTuple{nEdgesOnEdge, float_type(mesh)}}(undef, mesh.edges.n), indices.length)
    return compute_weightsOnEdge_Peixoto!(indices, weights, mesh)
end

function TangentialVelocityReconstructionPeixoto(mesh::AbstractVoronoiMesh)
    return TangentialVelocityReconstructionPeixoto(compute_weightsOnEdge_Peixoto(mesh)...)
end
