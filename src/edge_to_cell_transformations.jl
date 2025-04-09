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

struct EdgeToCellLinearInterpolation{NEdges, TI, TF} <: EdgeToCellTransformation
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

            MpM = qr!(M'*M)

            wvec = vec((inv(MpM)*M')[1,:])

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
            MpM = qr!(M'*M)

            wvec = vec((inv(MpM)*M')[1,:])

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

EdgeToCellLinearInterpolation(cells::Cells, edges::Edges) = EdgeToCellLinearInterpolation(edges.n, cells.edges, compute_weights_edge_to_cell_linear_interpolation(cells, edges))

EdgeToCellLinearInterpolation(mesh::AbstractVoronoiMesh) = EdgeToCellLinearInterpolation(mesh.cells, mesh.edges)

#struct EdgeToCellQuadraticInterpolation{NEdges, TI, TF} <: EdgeToCellTransformation
#    n::Int
#    indices::ImVecArray{NEdges, TI, 1}
#    weights::ImVecArray{NEdges, TF, 1}
#end
#
#function compute_weights_edge_to_cell_quadratic_interpolation_periodic!(w::ImVecArray{NE, TF, 1}, cellPos, edgePos, edgesOnCell, xp::Number, yp::Number) where {NE, TF}
#
#    wdata = w.data
#
#    @parallel for c in eachindex(cellPos)
#        @inbounds begin
#
#            cpos = cellPos[c]
#
#            eoc = edgesOnCell[c]
#            nEdges = length(eoc)
#
#            is_pentagon = nEdges == 5
#
#            nfields = is_pentagon ? 5 : 6
#
#            M = Matrix{TF}(undef, nEdges, nfields)
#
#            for ei in Base.OneTo(nEdges)
#                M[ei, 1] = oneunit(TF)
#                e = eoc[ei]
#                epos = closest(cpos, edgePos[e], xp , yp)
#                dpos = epos - cpos
#                xyterm = dpos.x*dpos.y / 3
#                M[ei, 2] = dpos.x
#                M[ei, 3] = dpos.y
#                M[ei, 4] = dpos.x^2 + dpos.y^2 / 2 + xyterm
#                M[ei, 5] = dpos.y^2 + xyterm
#                if !is_pentagon
#                    M[ei, 6] = xyterm
#                end
#            end
#
#            MpM = M'*M
#
#            wvec = vec((inv(MpM)*M')[1,:])
#
#            wdata[c] = padwith(ImmutableVector{NE, TF}(wvec), zero(TF)).data
#        end
#    end
#
#    return w
#end
#
#function compute_weights_edge_to_cell_quadratic_interpolation(cellPos::Vec2DxyArray{TF}, edgePos, edgesOnCell::ImVecArray{NE, TI, 1}, xp::Number, yp::Number) where {NE, TI, TF}
#    w = ImmutableVectorArray(Vector{NTuple{NE,TF}}(undef, length(cellPos)), edgesOnCell.length)
#    return compute_weights_edge_to_cell_quadratic_interpolation_periodic!(w, cellPos, edgePos, edgesOnCell, xp, yp)
#end
#
#compute_weights_edge_to_cell_quadratic_interpolation(cells::Cells{false}, edges::Edges{false}) = compute_weights_edge_to_cell_quadratic_interpolation(cells.position, edges.position, cells.edges, cells.x_period, cells.y_period)
#
#function compute_weights_edge_to_cell_quadratic_interpolation_spherical!(w::ImVecArray{NE, TF, 1}, cellPos, edgePos, edgesOnCell, R::Number) where {NE, TF}
#
#    wdata = w.data
#
#    @parallel for c in eachindex(cellPos)
#        @inbounds begin
#
#            cpos = cellPos[c] / R
#
#            eoc = edgesOnCell[c]
#            nEdges = length(eoc)
#
#            is_pentagon = nEdges == 5
#
#            nfields = is_pentagon ? 5 : 6
#
#            M = Matrix{TF}(undef, nEdges, nfields)
#
#            xdir = zero(eltype(cellPos))
#            ydir = zero(eltype(cellPos))
#
#            for ei in Base.OneTo(nEdges)
#                M[ei, 1] = oneunit(TF)
#
#                e = eoc[ei]
#                epos = edgePos[e] / R
#                ep_proj = epos * inv(epos ⋅ cpos)
#                dpos = ep_proj - cpos
#                if ei == 1
#                    aux = norm(dpos)
#                    xdir = dpos / aux
#                    ydir = normalize(cpos × xdir)
#                    M[ei, 2] = aux
#                    M[ei, 3] = zero(TF)
#                    M[ei, 4] = aux^2
#                    M[ei, 5] = zero(TF)
#                    if !is_pentagon
#                        M[ei, 6] = zero(TF)
#                    end
#                else
#                    dx = dpos ⋅ xdir
#                    dy = dpos ⋅ ydir
#                    dxdy = dx*dy
#                    M[ei, 2] = dx
#                    M[ei, 3] = dy
#                    M[ei, 4] = dx^2 + dy^2 / 2 + dxdy / 3
#                    M[ei, 5] = dy^2 + dxdy / 3
#                    if !is_pentagon
#                        M[ei, 6] = dxdy
#                    end
#                end
#            end
#
#            MpM = M'*M
#
#            wvec = vec((inv(MpM)*M')[1,:])
#
#            wdata[c] = padwith(ImmutableVector{NE, TF}(wvec), zero(TF)).data
#        end
#    end
#
#    return w
#end
#
#function compute_weights_edge_to_cell_quadratic_interpolation(cellPos::Vec3DArray{TF}, edgePos, edgesOnCell::ImVecArray{NE, TI, 1}, R::Number) where {NE, TI, TF}
#    w = ImmutableVectorArray(Vector{NTuple{NE,TF}}(undef, length(cellPos)), edgesOnCell.length)
#    return compute_weights_edge_to_cell_quadratic_interpolation_spherical!(w, cellPos, edgePos, edgesOnCell, R)
#end
#
#compute_weights_edge_to_cell_quadratic_interpolation(cells::Cells{true}, edges::Edges{true}) = compute_weights_edge_to_cell_quadratic_interpolation(cells.position, edges.position, cells.edges, cells.sphere_radius)
#
#compute_weights_edge_to_cell_quadratic_interpolation(mesh::AbstractVoronoiMesh) = compute_weights_edge_to_cell_quadratic_interpolation(mesh.cells, mesh.edges)
#
#EdgeToCellQuadraticInterpolation(cells::Cells, edges::Edges) = EdgeToCellQuadraticInterpolation(edges.n, cells.edges, compute_weights_edge_to_cell_quadratic_interpolation(cells, edges))
#
#EdgeToCellQuadraticInterpolation(mesh::AbstractVoronoiMesh) = EdgeToCellQuadraticInterpolation(mesh.cells, mesh.edges)

