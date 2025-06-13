abstract type EdgeToCellTransformation{NEdges, TI, TF} <: LinearVoronoiOperator end

name_input(::EdgeToCellTransformation) = "edge"
name_output(::EdgeToCellTransformation) = "cell"

struct EdgeToCellRingler{NEdges, TI, TF} <: EdgeToCellTransformation{NEdges, TI, TF}
    n::Int
    indices::ImVecArray{NEdges, TI, 1}
    weights::ImVecArray{NEdges, TF, 1}
end

function compute_weights_edge_to_cell_ringler!(w::ImVecArray{NE, TF, 1}, areaCell, dcEdge, dvEdge, edgesOnCell) where {NE, TF}

    wdata = w.data

    @batch for c in eachindex(areaCell)
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

struct EdgeToCellArea{NEdges, TI, TF} <: EdgeToCellTransformation{NEdges, TI, TF}
    n::Int
    indices::ImVecArray{NEdges, TI, 1}
    weights::ImVecArray{NEdges, TF, 1}
end

EdgeToCellArea(cells::Cells{false}, edges::Edges{false}, ::Vertices{false}) = EdgeToCellArea(edges.n, cells.edges, compute_weights_edge_to_cell_ringler(cells, edges))

function compute_weights_edge_to_cell_area!(w::ImVecArray{NE, TI, 1}, areaCell::AbstractVector{TF}, cellPos, vertexPos, verticesOnCell, R::Number)  where {NE, TI, TF}
    wdata = w.data

    @batch for c in eachindex(areaCell)
        @inbounds begin

            aux = ImmutableVector{NE,TF}()
            A = areaCell[c]
            c_pos = cellPos[c]

            voc =verticesOnCell[c]
            l = length(voc)

            #Assuming vertex[n] is between edge[n] and edge[n+1]
            prev_v_pos = vertexPos[voc[l]]
            for e in Base.OneTo(l)
                next_v_pos = vertexPos[voc[e]]
                aux = push(aux, spherical_polygon_area(R, c_pos, prev_v_pos, next_v_pos) / A)
                prev_v_pos = next_v_pos
            end

            #fix any float point errors (aux should sum to 1)
            aux = aux .+ ((1 - sum(aux)) / l)

            wdata[c] = padwith(aux, zero(TF)).data
        end
    end

    return w
end

function compute_weights_edge_to_cell_area(areaCell::AbstractVector{TF}, cellPos, vertexPos, verticesOnCell::ImVecArray{NE, TI, 1}, R::Number) where {NE, TI, TF}
    w = ImmutableVectorArray(Vector{NTuple{NE,TF}}(undef, length(areaCell)), verticesOnCell.length)
    return compute_weights_edge_to_cell_area!(w, areaCell, cellPos, vertexPos, verticesOnCell, R)
end

compute_weights_edge_to_cell_area(cells::Cells{true}, ::Edges{true}, vertices::Vertices{true}) =
    compute_weights_edge_to_cell_area(cells.area, cells.position, vertices.position, cells.vertices, cells.sphere_radius)

EdgeToCellArea(cells::Cells{true}, edges::Edges{true}, vertices::Vertices{true}) = EdgeToCellArea(edges.n, cells.edges, compute_weights_edge_to_cell_area(cells, edges, vertices))

EdgeToCellArea(mesh::AbstractVoronoiMesh) = EdgeToCellArea(mesh.cells, mesh.edges, mesh.vertices)

struct EdgeToCellLSq2{NEdges, TI, TF} <: EdgeToCellTransformation{NEdges, TI, TF}
    n::Int
    indices::ImVecArray{NEdges, TI, 1}
    weights::ImVecArray{NEdges, TF, 1}
end

compute_weights_edge_to_cell_linear_interpolation(cells::Cells{false}, edges::Edges{false}) = compute_weights_lsq(cells.position, edges.position, cells.edges, cells.x_period, cells.y_period, compute_weights_lsq2)

compute_weights_edge_to_cell_linear_interpolation(cells::Cells{true}, edges::Edges{true}) = compute_weights_lsq(cells.position, edges.position, cells.edges, cells.sphere_radius, compute_weights_lsq2)

EdgeToCellLSq2(cells::Cells, edges::Edges) = EdgeToCellLSq2(edges.n, cells.edges, compute_weights_edge_to_cell_linear_interpolation(cells, edges))

EdgeToCellLSq2(mesh::AbstractVoronoiMesh) = EdgeToCellLSq2(mesh.cells, mesh.edges)

struct EdgeToCellLSq3{NEdges, TI, TF} <: EdgeToCellTransformation{NEdges, TI, TF}
    n::Int
    indices::ImVecArray{NEdges, TI, 1}
    weights::ImVecArray{NEdges, TF, 1}
end

compute_weights_edge_to_cell_quadratic_interpolation(cells::Cells{false}, edges::Edges{false}) = compute_weights_lsq(cells.position, edges.position, cells.edges, cells.x_period, cells.y_period, compute_weights_lsq3)

compute_weights_edge_to_cell_quadratic_interpolation(cells::Cells{true}, edges::Edges{true}) = compute_weights_lsq(cells.position, edges.position, cells.edges, cells.sphere_radius, compute_weights_lsq3)

EdgeToCellLSq3(cells::Cells, edges::Edges) = EdgeToCellLSq3(edges.n, cells.edges, compute_weights_edge_to_cell_quadratic_interpolation(cells, edges))

EdgeToCellLSq3(mesh::AbstractVoronoiMesh) = EdgeToCellLSq3(mesh.cells, mesh.edges)

