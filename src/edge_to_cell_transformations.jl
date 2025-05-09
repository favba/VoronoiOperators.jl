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

struct EdgeToCellArea{NEdges, TI, TF} <: EdgeToCellTransformation{NEdges, TI, TF}
    n::Int
    indices::ImVecArray{NEdges, TI, 1}
    weights::ImVecArray{NEdges, TF, 1}
end

EdgeToCellArea(cells::Cells{false}, edges::Edges{false}, ::Vertices{false}) = EdgeToCellArea(edges.n, cells.edges, compute_weights_edge_to_cell_ringler(cells, edges))

function compute_weights_edge_to_cell_area!(w::ImVecArray{NE, TI, 1}, areaCell::AbstractVector{TF}, cellPos, vertexPos, verticesOnCell, R::Number)  where {NE, TI, TF}
    wdata = w.data

    @parallel for c in eachindex(areaCell)
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

function compute_weights_edge_to_cell_linear_interpolation_periodic!(w::ImVecArray{NE, TF, 1}, cellPos, edgePos, edgesOnCell, xp::Number, yp::Number) where {NE, TF}

    wdata = w.data

    @parallel for c in eachindex(cellPos)
        @inbounds begin

            cpos = cellPos[c]

            eoc = edgesOnCell[c]
            nEdges = length(eoc)

            M = Matrix{TF}(undef, nEdges, 3)

            for ei in Base.OneTo(nEdges)
                M[ei, 1] = oneunit(TF)
                e = eoc[ei]
                epos = closest(cpos, edgePos[e], xp , yp)
                dpos = epos - cpos
                M[ei, 2] = dpos.x
                M[ei, 3] = dpos.y
            end

            MpM = cholesky!(Hermitian(M'*M))

            wvec = vec((LinearAlgebra.inv!(MpM)*M')[1,:])

            wdata[c] = padwith(ImmutableVector{NE, TF}(wvec), zero(TF)).data
        end
    end

    return w
end

function compute_weights_edge_to_cell_linear_interpolation(cellPos::Vec2DxyArray{TF}, edgePos, edgesOnCell::ImVecArray{NE, TI, 1}, xp::Number, yp::Number) where {NE, TI, TF}
    w = ImmutableVectorArray(Vector{NTuple{NE,TF}}(undef, length(cellPos)), edgesOnCell.length)
    return compute_weights_edge_to_cell_linear_interpolation_periodic!(w, cellPos, edgePos, edgesOnCell, xp, yp)
end

compute_weights_edge_to_cell_linear_interpolation(cells::Cells{false}, edges::Edges{false}) = compute_weights_edge_to_cell_linear_interpolation(cells.position, edges.position, cells.edges, cells.x_period, cells.y_period)

function compute_weights_edge_to_cell_linear_interpolation_spherical!(w::ImVecArray{NE, TF, 1}, cellPos, edgePos, edgesOnCell, R::Number) where {NE, TF}

    wdata = w.data

    @parallel for c in eachindex(cellPos)
        @inbounds begin

            cpos = cellPos[c] / R

            eoc = edgesOnCell[c]
            nEdges = length(eoc)

            M = Matrix{TF}(undef, nEdges, 3)

            xdir = zero(eltype(cellPos))
            ydir = zero(eltype(cellPos))

            for ei in Base.OneTo(nEdges)
                M[ei, 1] = oneunit(TF)

                e = eoc[ei]
                epos = edgePos[e] / R
                ep_proj = epos * inv(epos ⋅ cpos)
                dpos = ep_proj - cpos
                if ei == 1
                    aux = norm(dpos)
                    xdir = dpos / aux
                    ydir = normalize(cpos × xdir)
                    M[ei, 2] = aux
                    M[ei, 3] = zero(TF)
                else
                    M[ei, 2] = dpos ⋅ xdir
                    M[ei, 3] = dpos ⋅ ydir
                end
            end
            MpM = cholesky!(Hermitian(M'*M))

            wvec = vec((LinearAlgebra.inv!(MpM)*M')[1,:])

            wdata[c] = padwith(ImmutableVector{NE, TF}(wvec), zero(TF)).data
        end
    end

    return w
end

function compute_weights_edge_to_cell_linear_interpolation(cellPos::Vec3DArray{TF}, edgePos, edgesOnCell::ImVecArray{NE, TI, 1}, R::Number) where {NE, TI, TF}
    w = ImmutableVectorArray(Vector{NTuple{NE,TF}}(undef, length(cellPos)), edgesOnCell.length)
    return compute_weights_edge_to_cell_linear_interpolation_spherical!(w, cellPos, edgePos, edgesOnCell, R)
end

compute_weights_edge_to_cell_linear_interpolation(cells::Cells{true}, edges::Edges{true}) = compute_weights_edge_to_cell_linear_interpolation(cells.position, edges.position, cells.edges, cells.sphere_radius)

compute_weights_edge_to_cell_linear_interpolation(mesh::AbstractVoronoiMesh) = compute_weights_edge_to_cell_linear_interpolation(mesh.cells, mesh.edges)

EdgeToCellLSq2(cells::Cells, edges::Edges) = EdgeToCellLSq2(edges.n, cells.edges, compute_weights_edge_to_cell_linear_interpolation(cells, edges))

EdgeToCellLSq2(mesh::AbstractVoronoiMesh) = EdgeToCellLSq2(mesh.cells, mesh.edges)

struct EdgeToCellLSq3{NEdges, TI, TF} <: EdgeToCellTransformation{NEdges, TI, TF}
    n::Int
    indices::ImVecArray{NEdges, TI, 1}
    weights::ImVecArray{NEdges, TF, 1}
end

function compute_weights_edge_to_cell_quadratic_interpolation_periodic!(w::ImVecArray{NE, TF, 1}, cellPos, edgePos, edgesOnCell, xp::Number, yp::Number) where {NE, TF}

    wdata = w.data

    @parallel for c in eachindex(cellPos)
        @inbounds begin

            cpos = cellPos[c]

            eoc = edgesOnCell[c]
            nEdges = length(eoc)

            M = Matrix{TF}(undef, nEdges, 6)

            for ei in Base.OneTo(nEdges)
                e = eoc[ei]
                epos = closest(cpos, edgePos[e], xp , yp)
                dpos = epos - cpos
                xyterm = dpos.x*dpos.y / 3
                M[ei, 1] = oneunit(TF)
                M[ei, 2] = dpos.x
                M[ei, 3] = dpos.y
                M[ei, 4] = dpos.x^2
                M[ei, 5] = dpos.y^2
                M[ei, 6] = xyterm
            end

            MpM = Hermitian(M'*M)

            #Perform regularization if condition number is too big (or Inf)
            #Regularize only quadratic terms, to preserve 1st order and 2nd order terms
            cn = cond(MpM)
            ii = 0
            while cn > 5e6
                ii += 1
                #@show c, ii, cn
                for i in 4:6
                    MpM[i, i] += 1e-6
                end
                cn = cond(MpM)
            end

            wvec = vec((LinearAlgebra.inv!(cholesky!(MpM))*M')[1,:])

            wdata[c] = padwith(ImmutableVector{NE, TF}(wvec), zero(TF)).data
        end
    end

    return w
end

function compute_weights_edge_to_cell_quadratic_interpolation(cellPos::Vec2DxyArray{TF}, edgePos, edgesOnCell::ImVecArray{NE, TI, 1}, xp::Number, yp::Number) where {NE, TI, TF}
    w = ImmutableVectorArray(Vector{NTuple{NE,TF}}(undef, length(cellPos)), edgesOnCell.length)
    return compute_weights_edge_to_cell_quadratic_interpolation_periodic!(w, cellPos, edgePos, edgesOnCell, xp, yp)
end

compute_weights_edge_to_cell_quadratic_interpolation(cells::Cells{false}, edges::Edges{false}) = compute_weights_edge_to_cell_quadratic_interpolation(cells.position, edges.position, cells.edges, cells.x_period, cells.y_period)

function compute_weights_edge_to_cell_quadratic_interpolation_spherical!(w::ImVecArray{NE, TF, 1}, cellPos, edgePos, edgesOnCell, R::Number) where {NE, TF}

    wdata = w.data

    @parallel for c in eachindex(cellPos)
        @inbounds begin

            cpos = cellPos[c] / R

            eoc = edgesOnCell[c]
            nEdges = length(eoc)

            M = Matrix{TF}(undef, nEdges, 6)

            xdir = zero(eltype(cellPos))
            ydir = zero(eltype(cellPos))

            for ei in Base.OneTo(nEdges)
                M[ei, 1] = oneunit(TF)

                e = eoc[ei]
                epos = edgePos[e] / R
                ep_proj = epos * inv(epos ⋅ cpos)
                dpos = ep_proj - cpos
                if ei == 1
                    aux = norm(dpos)
                    xdir = dpos / aux
                    ydir = normalize(cpos × xdir)
                    M[ei, 2] = aux
                    M[ei, 3] = zero(TF)
                    M[ei, 4] = aux^2
                    M[ei, 5] = zero(TF)
                    M[ei, 6] = zero(TF)
                else
                    dx = dpos ⋅ xdir
                    dy = dpos ⋅ ydir
                    dxdy = dx*dy
                    M[ei, 2] = dx
                    M[ei, 3] = dy
                    M[ei, 4] = dx^2
                    M[ei, 5] = dy^2
                    M[ei, 6] = dxdy
                end
            end

            MpM = Hermitian(M'*M)
            cn = cond(MpM)
            ii = 0
            while cn > 5e6
                ii += 1
                #@show c, ii, cn
                for i in 4:6
                    MpM[i, i] += 1e-6
                end
                cn = cond(MpM)
            end

            wvec = vec((LinearAlgebra.inv!(cholesky!(MpM))*M')[1,:])

            wdata[c] = padwith(ImmutableVector{NE, TF}(wvec), zero(TF)).data
        end
    end

    return w
end

function compute_weights_edge_to_cell_quadratic_interpolation(cellPos::Vec3DArray{TF}, edgePos, edgesOnCell::ImVecArray{NE, TI, 1}, R::Number) where {NE, TI, TF}
    w = ImmutableVectorArray(Vector{NTuple{NE,TF}}(undef, length(cellPos)), edgesOnCell.length)
    return compute_weights_edge_to_cell_quadratic_interpolation_spherical!(w, cellPos, edgePos, edgesOnCell, R)
end

compute_weights_edge_to_cell_quadratic_interpolation(cells::Cells{true}, edges::Edges{true}) = compute_weights_edge_to_cell_quadratic_interpolation(cells.position, edges.position, cells.edges, cells.sphere_radius)

compute_weights_edge_to_cell_quadratic_interpolation(mesh::AbstractVoronoiMesh) = compute_weights_edge_to_cell_quadratic_interpolation(mesh.cells, mesh.edges)

EdgeToCellLSq3(cells::Cells, edges::Edges) = EdgeToCellLSq3(edges.n, cells.edges, compute_weights_edge_to_cell_quadratic_interpolation(cells, edges))

EdgeToCellLSq3(mesh::AbstractVoronoiMesh) = EdgeToCellLSq3(mesh.cells, mesh.edges)

