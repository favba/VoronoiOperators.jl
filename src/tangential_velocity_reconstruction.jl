abstract type TangentialVelocityReconstruction{N_MAX, TF, TI} <: LinearVoronoiOperator end
name_input(::TangentialVelocityReconstruction) = "edge"
name_output(::TangentialVelocityReconstruction) = "edge"
n_input(a::TangentialVelocityReconstruction) = n_output(a)

struct TangentialVelocityReconstructionThuburn{N_MAX, TF, TI} <: TangentialVelocityReconstruction{N_MAX, TF, TI}
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
    return compute_weightsOnEdge_trisk!(indices, w, edges.cells, cells.vertices, cells.edges, edges.cellsDistance, edges.length, vertices.kiteAreas, vertices.cells, cells.nEdges, cells.area)
end

compute_weightsOnEdge_trisk!(indices, weightsOnEdge,mesh::VoronoiMesh) = compute_weightsOnEdge_trisk!(indices, weightsOnEdge, mesh.cells, mesh.vertices, mesh.edges)

function compute_weightsOnEdge_trisk(mesh::VoronoiMesh)
    nEdgesOnEdge = maximum(x->(x[1] + x[2] - 2), ((a,inds) -> @inbounds((a[inds[1]], a[inds[2]]))).((mesh.cells.nEdges,), mesh.edges.cells))
    indices =ImVecArray{nEdgesOnEdge, integer_type(mesh)}(mesh.edges.n)
    weights =ImmutableVectorArray(Vector{NTuple{nEdgesOnEdge, float_type(mesh)}}(undef, mesh.edges.n), indices.length)
    return compute_weightsOnEdge_trisk!(indices, weights, mesh)
end

function TangentialVelocityReconstructionThuburn(mesh::VoronoiMesh)
    return TangentialVelocityReconstructionThuburn(compute_weightsOnEdge_trisk(mesh)...)
end
