abstract type EdgeToCellTransformation <: LinearVoronoiOperator end

name_input(::EdgeToCellTransformation) = "edge"
name_output(::EdgeToCellTransformation) = "cell"

struct EdgeToCellRingler{NEdges, TI, TF} <: EdgeToCellTransformation
    n::Int
    indices::ImVecArray{NEdges, TI, 1}
    weights::ImVecArray{NEdges, TF, 1}
end

function compute_weights_edge_to_cell_ringler!(w::ImVecArray{NE, TF, 1}, areaCell, dcEdge, dvEdge, edgesOnCell) where {NE, TF}

    wdata = w.data

    @parallel for c in eachindex(areaCell)
        @inbounds begin

            aux = ImmutableVector{NE,TF}()
            term = inv(4 * areaCell[c])

            for e in edgesOnCell[c]
                aux = push(aux, term * dcEdge[e] * dvEdge[e])
            end

            wdata[c] = padwith(aux, zero(TF)).data
        end
    end

    return w
end

function compute_weights_edge_to_cell_ringler(areaCell::AbstractVector{TF}, dcEdge, dvEdge, edgesOnCell::ImVecArray{NE, TI, 1}) where {NE, TI, TF}
    w = ImmutableVectorArray(Vector{NTuple{NE,TF}}(undef, length(areaCell)), edgesOnCell.length)
    return compute_weights_edge_to_cell_ringler!(w, areaCell, dcEdge, dvEdge, edgesOnCell)
end

compute_weights_edge_to_cell_ringler(cells::Cells, edges::Edges) = compute_weights_edge_to_cell_ringler(cells.area, edges.lengthDual, edges.length, cells.edges)

compute_weights_edge_to_cell_ringler(mesh::AbstractVoronoiMesh) = compute_weights_edge_to_cell_ringler(mesh.cells, mesh.edges)

EdgeToCellRingler(cells::Cells, edges::Edges) = EdgeToCellRingler(edges.n, cells.edges, compute_weights_edge_to_cell_ringler(cells, edges))

EdgeToCellRingler(mesh::AbstractVoronoiMesh) = EdgeToCellRingler(mesh.cells, mesh.edges)

struct EdgeToCellInterpolationLinear{NEdges, TI, TF} <: EdgeToCellTransformation
    n::Int
    indices::ImVecArray{NEdges, TI, 1}
    weights::ImVecArray{NEdges, TF, 1}
end
