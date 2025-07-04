abstract type DifferentialOperator <: LinearVoronoiOperator end

struct GradientAtEdge{TI, TF} <: DifferentialOperator
    n::Int
    dc::Vector{TF}
    indices::Vector{FixedVector{2, TI}}
end

name_input(::GradientAtEdge) = "cell"
name_output(::GradientAtEdge) = "edge"
out_eltype(Vop::GradientAtEdge, in_field, op::F = Base.identity) where {F} = Base.promote_op(*, eltype(eltype(Vop.dc)), Base.promote_op(op, eltype(in_field)))

GradientAtEdge(mesh::AbstractVoronoiMesh) = GradientAtEdge(mesh.cells.n, mesh.edges.lengthDual, mesh.edges.cells)

function gradient_at_edge!(out::AbstractVector, c_field, dc, cellsOnEdge, op::F = Base.identity) where {F <: Function}
    @batch for e in eachindex(cellsOnEdge)
        @inbounds @inline begin
            c1, c2 = cellsOnEdge[e]
            out[e] = (op(c_field[c2]) - op(c_field[c1])) / dc[e]
        end
    end
    return out
end

function gradient_at_edge!(out::AbstractVector, op_out::F, c_field, dc, cellsOnEdge, op::F2 = Base.identity) where {F <: Function, F2 <: Function}
    @batch for e in eachindex(cellsOnEdge)
        @inbounds @inline begin
            c1, c2 = cellsOnEdge[e]
            out[e] = op_out(out[e], (op(c_field[c2]) - op(c_field[c1])) / dc[e])
        end
    end
    return out
end

function gradient_at_edge!(out::AbstractMatrix{T}, c_field, dc, cellsOnEdge, op::F = Base.identity) where {T, F <: Function}

    N_SIMD = simd_length(T)
    ValN_SIMD = Val{N_SIMD}()
    lane = sd.VecRange{N_SIMD}(0)
    Nk = size(out, 1)

    range_simd, range_serial = simd_ranges(N_SIMD, Nk)
    is_there_rest = length(range_serial) != 0
    k_simd_end = lane + (Nk - N_SIMD + 1)

    @batch for e in eachindex(cellsOnEdge)
        @inbounds @inline begin
            c1, c2 = cellsOnEdge[e]
            inv_dc = inv(dc[e])
            inv_dc_simd = simd_repeat(ValN_SIMD, inv_dc)

            for k in range_simd
                k_simd = lane + k
                out[k_simd, e] = inv_dc_simd * (op(c_field[k_simd, c2]) - op(c_field[k_simd, c1]))
            end

            #for k in range_serial
            #    out[k, e] = inv_dc * (op(c_field[k, c2]) - op(c_field[k, c1]))
            #end
            if is_there_rest
                out[k_simd_end, e] = inv_dc_simd * (op(c_field[k_simd_end, c2]) - op(c_field[k_simd_end, c1]))
            end

        end #inbounds
    end
    return out
end

function gradient_at_edge!(out::AbstractMatrix{T}, op_out::F, c_field, dc, cellsOnEdge, op::F2 = Base.identity) where {T, F <: Function, F2 <: Function}

    N_SIMD = simd_length(T)
    ValN_SIMD = Val{N_SIMD}()
    lane = sd.VecRange{N_SIMD}(0)
    Nk = size(out, 1)

    range_simd, range_serial = simd_ranges(N_SIMD, Nk)

    @batch for e in eachindex(cellsOnEdge)
        @inbounds @inline begin
            c1, c2 = cellsOnEdge[e]
            inv_dc = inv(dc[e])
            inv_dc_simd = simd_repeat(ValN_SIMD, inv_dc)

            for k in range_simd
                k_simd = lane + k
                out[k_simd, e] = op_out(out[k_simd, e], inv_dc_simd * (op(c_field[k_simd, c2]) - op(c_field[k_simd, c1])))
            end

            for k in range_serial
                out[k, e] = op_out(out[k, e], inv_dc * (op(c_field[k, c2]) - op(c_field[k, c1])))
            end

        end #inbounds
    end

    return out
end

function gradient_at_edge!(out::AbstractMatrix{T}, op_out::typeof(+), c_field, dc, cellsOnEdge, op::F = Base.identity) where {T, F <: Function}

    N_SIMD = simd_length(T)
    ValN_SIMD = Val{N_SIMD}()
    lane = sd.VecRange{N_SIMD}(0)
    Nk = size(out, 1)

    range_simd, range_serial = simd_ranges(N_SIMD, Nk)

    @batch for e in eachindex(cellsOnEdge)
        @inbounds @inline begin
            c1, c2 = cellsOnEdge[e]
            inv_dc = inv(dc[e])
            inv_dc_simd = simd_repeat(ValN_SIMD, inv_dc)

            for k in range_simd
                k_simd = lane + k
                out[k_simd, e] = muladd(inv_dc_simd, (op(c_field[k_simd, c2]) - op(c_field[k_simd, c1])), out[k_simd, e])
            end

            for k in range_serial
                out[k, e] = muladd(inv_dc, (op(c_field[k, c2]) - op(c_field[k, c1])), out[k, e])
            end

        end #inbounds
    end

    return out
end

function gradient_at_edge!(out::AbstractArray{T, 3}, c_field, dc, cellsOnEdge, op::F = Base.identity) where {T, F <: Function}

    N_SIMD = simd_length(T)
    ValN_SIMD = Val{N_SIMD}()
    lane = sd.VecRange{N_SIMD}(0)
    Nk = size(out, 1)

    range_simd, range_serial = simd_ranges(N_SIMD, Nk)
    is_there_rest = length(range_serial) != 0
    k_simd_end = lane + (Nk - N_SIMD + 1)

    @batch for e in eachindex(cellsOnEdge)
        @inbounds @inline begin
            c1, c2 = cellsOnEdge[e]
            inv_dc = inv(dc[e])
            inv_dc_simd = simd_repeat(ValN_SIMD, inv_dc)

            for t in axes(out, 3)

                for k in range_simd
                    k_simd = lane + k
                    out[k_simd, e, t] = inv_dc_simd * (op(c_field[k_simd, c2, t]) - op(c_field[k_simd, c1, t]))
                end

                #for k in range_serial
                #    out[k, e, t] = inv_dc * (op(c_field[k, c2, t]) - op(c_field[k, c1, t]))
                #end
                if is_there_rest
                    out[k_simd_end, e, t] = inv_dc_simd * (op(c_field[k_simd_end, c2, t]) - op(c_field[k_simd_end, c1, t]))
                end
            end
        end #inbounds
    end
    return out
end

function gradient_at_edge!(out::AbstractArray{T, 3}, op_out::F, c_field, dc, cellsOnEdge, op::F2 = Base.identity) where {T, F <: Function, F2 <: Function}

    N_SIMD = simd_length(T)
    ValN_SIMD = Val{N_SIMD}()
    lane = sd.VecRange{N_SIMD}(0)
    Nk = size(out, 1)

    range_simd, range_serial = simd_ranges(N_SIMD, Nk)

    @batch for e in eachindex(cellsOnEdge)
        @inbounds @inline begin
            c1, c2 = cellsOnEdge[e]
            inv_dc = inv(dc[e])
            inv_dc_simd = simd_repeat(ValN_SIMD, inv_dc)

            for t in axes(out, 3)

                for k in range_simd
                    k_simd = lane + k
                    out[k_simd, e, t] = op_out(out[k_simd, e, t], inv_dc_simd * (op(c_field[k_simd, c2, t]) - op(c_field[k_simd, c1, t])))
                end

                for k in range_serial
                    out[k, e, t] = op_out(out[k, e, t], inv_dc * (op(c_field[k, c2, t]) - op(c_field[k, c1, t])))
                end
            end
        end #inbounds
    end

    return out
end

function gradient_at_edge!(out::AbstractArray{T, 3}, op_out::typeof(+), c_field, dc, cellsOnEdge, op::F = Base.identity) where {T, F <: Function}

    N_SIMD = simd_length(T)
    ValN_SIMD = Val{N_SIMD}()
    lane = sd.VecRange{N_SIMD}(0)
    Nk = size(out, 1)

    range_simd, range_serial = simd_ranges(N_SIMD, Nk)

    @batch for e in eachindex(cellsOnEdge)
        @inbounds @inline begin
            c1, c2 = cellsOnEdge[e]
            inv_dc = inv(dc[e])
            inv_dc_simd = simd_repeat(ValN_SIMD, inv_dc)

            for t in axes(out, 3)

                for k in range_simd
                    k_simd = lane + k
                    out[k_simd, e, t] = muladd(inv_dc_simd, (op(c_field[k_simd, c2, t]) - op(c_field[k_simd, c1, t])), out[k_simd, e, t])
                end

                for k in range_serial
                    out[k, e, t] = muladd(inv_dc, (op(c_field[k, c2, t]) - op(c_field[k, c1, t])), out[k, e, t])
                end
            end
        end #inbounds
    end

    return out
end

function (Vop::GradientAtEdge)(out_field::AbstractArray, in_field::AbstractArray, op::F = Base.identity) where {F <: Function}
    is_proper_size(in_field, n_input(Vop)) || throw(DimensionMismatch("Input array doesn't seem to be a $(name_input(Vop)) field"))
    is_proper_size(out_field, n_output(Vop)) || throw(DimensionMismatch("Output array doesn't seem to be a $(name_output(Vop)) field"))

    gradient_at_edge!(out_field, in_field, Vop.dc, Vop.indices, op)

    return out_field
end

function (Vop::GradientAtEdge)(out_field::AbstractArray, op_out::F, in_field::AbstractArray, op::F2 = Base.identity) where {F <: Function, F2 <: Function}
    is_proper_size(in_field, n_input(Vop)) || throw(DimensionMismatch("Input array doesn't seem to be a $(name_input(Vop)) field"))
    is_proper_size(out_field, n_output(Vop)) || throw(DimensionMismatch("Output array doesn't seem to be a $(name_output(Vop)) field"))

    gradient_at_edge!(out_field, op_out, in_field, Vop.dc, Vop.indices, op)

    return out_field
end

abstract type CellDiv{TI, TF} <: DifferentialOperator end

struct DivAtCell{N_MAX, TI, TF} <: CellDiv{TI, TF}
    n::Int
    indices::SmVecArray{N_MAX, TI, 1}
    weights::SmVecArray{N_MAX, TF, 1}
end

name_input(::DivAtCell) = "edge"
name_output(::DivAtCell) = "cell"

function compute_div_at_cell_weights(areaCell, edgesOnCell::AbstractVector{<:SmallVector{N_MAX}}, dvEdge::AbstractVector{T}, cellsOnEdge) where {N_MAX, T}
    w = Vector{FixedVector{N_MAX, T}}(undef, length(areaCell))

    @batch for c in eachindex(areaCell)
        @inbounds begin
            inv_a = inv(areaCell[c])

            eoc = edgesOnCell[c]
            l = length(eoc)
            aux = SmallVector{N_MAX, T}()
            for i in Base.OneTo(l)
                e = eoc[i]
                Le = dvEdge[e]
                aux = push(aux, Le * inv_a * VoronoiMeshes.sign_edge(cellsOnEdge[e], c))
            end
            w[c] = fixedvector(aux)
        end
    end
    return SmallVectorArray(w, edgesOnCell.length)
end

function DivAtCell(mesh::AbstractVoronoiMesh)
    w = compute_div_at_cell_weights(mesh.cells.area, mesh.cells.edges, mesh.edges.length, mesh.edges.cells)
    return DivAtCell(mesh.edges.n, mesh.cells.edges, w)
end

struct CurlAtVertex{TI, TF} <: DifferentialOperator
    n::Int
    weights::Vector{FixedVector{3, TF}}
    indices::Vector{FixedVector{3, TI}}
end

name_input(::CurlAtVertex) = "edge"
name_output(::CurlAtVertex) = "vertex"

function compute_rotational_at_vertex_weights!(weights, areaVertex, dc, edgesOnVertex, cellsOnVertex, cellsOnEdge)

    @batch for v in eachindex(weights)
        e1, e2, e3 = edgesOnVertex[v]
        c1, c2, c3 = cellsOnVertex[v]

        c1e1, c2e1 = cellsOnEdge[e1]
        if (c3, c1) == (c1e1, c2e1)
            sign_e1 = 1
        elseif (c1, c3) == (c1e1, c2e1)
            sign_e1 = -1
        else
            error("Is the vertex info ordering correct?")
        end

        c1e2, c2e2 = cellsOnEdge[e2]
        if (c1, c2) == (c1e2, c2e2)
            sign_e2 = 1
        elseif (c2, c1) == (c1e2, c2e2)
            sign_e2 = -1
        else
            error("Is the vertex info ordering correct?")
        end

        c1e3, c2e3 = cellsOnEdge[e3]
        if (c2, c3) == (c1e3, c2e3)
            sign_e3 = 1
        elseif (c3, c2) == (c1e3, c2e3)
            sign_e3 = -1
        else
            error("Is the vertex info ordering correct?")
        end

        av = areaVertex[v]

        weights[v] = (sign_e1 * dc[e1], sign_e2 * dc[e2], sign_e3 * dc[e3]) ./ av
    end

    return weights
end

function compute_rotational_at_vertex_weights(areaVertex, dc, edgesOnVertex, cellsOnVertex, cellsOnEdge)
    weights = similar(areaVertex, FixedVector{3, Base.promote_op(/, eltype(dc), eltype(areaVertex))})
    return compute_rotational_at_vertex_weights!(weights, areaVertex, dc, edgesOnVertex, cellsOnVertex, cellsOnEdge)
end

CurlAtVertex(mesh::AbstractVoronoiMesh) = CurlAtVertex(mesh.edges.n, compute_rotational_at_vertex_weights(mesh.vertices.area, mesh.edges.lengthDual, mesh.vertices.edges, mesh.vertices.cells, mesh.edges.cells), mesh.vertices.edges)

struct CurlAtEdge{TI, TF} <: DifferentialOperator
    weights::Vector{FixedVector{4, TF}}
    indices::Vector{FixedVector{4, TI}}
end

n_input(a::CurlAtEdge) = n_output(a)

name_input(::CurlAtEdge) = "edge"
name_output(::CurlAtEdge) = "edge"

function find_other_edge_that_shares_cell(e::Integer, edges::FixedVector{3}, c::Integer, cellsOnEdge)
    i = findfirst(x -> ((x != e) && (c in cellsOnEdge[x])), edges)
    isnothing(i) && error("Couldn't find edge in $edges that share cell $c with $e.")
    return edges[i]
end

function compute_rotational_at_edge_weights!(weights, indices, areaVertex, dc, edgesOnVertex, cellsOnEdge, verticesOnEdge)

    @batch for e in eachindex(weights)
        v1, v2 = verticesOnEdge[e]
        c1, c2 = cellsOnEdge[e]

        A = areaVertex[v1] + areaVertex[v2]

        ee1 = find_other_edge_that_shares_cell(e, edgesOnVertex[v2], c1, cellsOnEdge)
        ee1sign = (c1 == cellsOnEdge[ee1][2]) ? 1 : -1
        w1 = ee1sign * dc[ee1] / A

        ee2 = find_other_edge_that_shares_cell(e, edgesOnVertex[v1], c1, cellsOnEdge)
        ee2sign = (c1 == cellsOnEdge[ee2][2]) ? -1 : 1
        w2 = ee2sign * dc[ee2] / A

        ee3 = find_other_edge_that_shares_cell(e, edgesOnVertex[v1], c2, cellsOnEdge)
        ee3sign = (c2 == cellsOnEdge[ee3][2]) ? 1 : -1
        w3 = ee3sign * dc[ee3] / A

        ee4 = find_other_edge_that_shares_cell(e, edgesOnVertex[v2], c2, cellsOnEdge)
        ee4sign = (c2 == cellsOnEdge[ee4][2]) ? -1 : 1
        w4 = ee4sign * dc[ee4] / A

        weights[e] = (w1, w2, w3, w4)
        indices[e] = (ee1, ee2, ee3, ee4)
    end

    return weights, indices
end

function compute_rotational_at_edge_weights(areaVertex, dc, edgesOnVertex, cellsOnEdge, verticesOnEdge)
    weights = similar(dc, FixedVector{4, Base.promote_op(/, eltype(dc), eltype(areaVertex))})
    indices = similar(dc, FixedVector{4, eltype(eltype(edgesOnVertex))})
    return compute_rotational_at_edge_weights!(weights, indices, areaVertex, dc, edgesOnVertex, cellsOnEdge, verticesOnEdge)
end

CurlAtEdge(mesh::AbstractVoronoiMesh) = CurlAtEdge(compute_rotational_at_edge_weights(mesh.vertices.area, mesh.edges.lengthDual, mesh.vertices.edges, mesh.edges.cells, mesh.edges.vertices)...)
