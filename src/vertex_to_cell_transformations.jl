abstract type VertexToCellTransformation{NEdges, TI, TF} <: LinearVoronoiOperator end

name_input(::VertexToCellTransformation) = "vertex"
name_output(::VertexToCellTransformation) = "cell"

struct VertexToCellArea{NEdges, TI, TF} <: VertexToCellTransformation{NEdges, TI, TF}
    n::Int
    indices::ImVecArray{NEdges, TI, 1}
    weights::ImVecArray{NEdges, TF, 1}
end

function compute_weights_vertex_to_cell_area!(w::ImVecArray{NE, TI, 1}, areaCell::AbstractVector{TF}, verticesOnCell, cellsOnVertex, kiteAreaOnVertex::AbstractVector{NTuple{3,TF}})  where {NE, TI, TF}
    wdata = w.data

    @batch for c in eachindex(areaCell)
        @inbounds begin

            aux = ImmutableVector{NE,TF}()
            A = areaCell[c]

            voc =verticesOnCell[c]
            l = length(voc)

            for v in Base.OneTo(l)
                aux = push(aux, VoronoiMeshes.select_kite_area(kiteAreaOnVertex, cellsOnVertex, voc[v], c) / A)
            end

            #fix any float point errors (aux should sum to 1)
            aux = aux .+ ((1 - sum(aux)) / l)

            wdata[c] = padwith(aux, zero(TF)).data
        end
    end

    return w
end

function compute_weights_vertex_to_cell_area(areaCell::AbstractVector{TF}, verticesOnCell::ImVecArray{NE, TI, 1}, cellsOnVertex, kiteAreaOnVertex) where {NE, TI, TF}
    w = ImmutableVectorArray(Vector{NTuple{NE,TF}}(undef, length(areaCell)), verticesOnCell.length)
    return compute_weights_vertex_to_cell_area!(w, areaCell, verticesOnCell, cellsOnVertex, kiteAreaOnVertex)
end

function compute_weights_vertex_to_cell_area(cells::Cells, vertices::Vertices)
    return compute_weights_vertex_to_cell_area(cells.area, cells.vertices, vertices.cells, vertices.kiteAreas)
end

VertexToCellArea(m::AbstractVoronoiMesh) = VertexToCellArea(m.vertices.n, m.cells.vertices, compute_weights_vertex_to_cell_area(m.cells, m.vertices))

struct VertexToCellLSq2{NEdges, TI, TF} <: VertexToCellTransformation{NEdges, TI, TF}
    n::Int
    indices::ImVecArray{NEdges, TI, 1}
    weights::ImVecArray{NEdges, TF, 1}
end

compute_weights_vertex_to_cell_linear_interpolation(cells::Cells{false}, vertices::Vertices{false}) = compute_weights_lsq(cells.position, vertices.position, cells.vertices, cells.x_period, cells.y_period, compute_weights_lsq2)

compute_weights_vertex_to_cell_linear_interpolation(cells::Cells{true}, vertices::Vertices{true}) = compute_weights_lsq(cells.position, vertices.position, cells.vertices, cells.sphere_radius, compute_weights_lsq2)

VertexToCellLSq2(cells::Cells, vertices::Vertices) = VertexToCellLSq2(vertices.n, cells.vertices, compute_weights_vertex_to_cell_linear_interpolation(cells, vertices))

VertexToCellLSq2(mesh::AbstractVoronoiMesh) = VertexToCellLSq2(mesh.cells, mesh.vertices)

struct VertexToCellLSq3{NEdges, TI, TF} <: VertexToCellTransformation{NEdges, TI, TF}
    n::Int
    indices::ImVecArray{NEdges, TI, 1}
    weights::ImVecArray{NEdges, TF, 1}
end

compute_weights_vertex_to_cell_quadratic_interpolation(cells::Cells{false}, vertices::Vertices{false}) = compute_weights_lsq(cells.position, vertices.position, cells.vertices, cells.x_period, cells.y_period, compute_weights_lsq3)

compute_weights_vertex_to_cell_quadratic_interpolation(cells::Cells{true}, vertices::Vertices{true}) = compute_weights_lsq(cells.position, vertices.position, cells.vertices, cells.sphere_radius, compute_weights_lsq3)

VertexToCellLSq3(cells::Cells, vertices::Vertices) = VertexToCellLSq3(vertices.n, cells.vertices, compute_weights_vertex_to_cell_quadratic_interpolation(cells, vertices))

VertexToCellLSq3(mesh::AbstractVoronoiMesh) = VertexToCellLSq3(mesh.cells, mesh.vertices)
