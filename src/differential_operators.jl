abstract type DifferentialOperator <: LinearVoronoiOperator end

struct GradientAtEdge{TI, TF} <: DifferentialOperator
    n::Int
    dc::Vector{TF}
    indices::Vector{NTuple{2, TI}}
end

name_input(::GradientAtEdge) = "cell"
name_output(::GradientAtEdge) = "edge"

GradientAtEdge(mesh::VoronoiMesh) = GradientAtEdge(mesh.cells.n, mesh.edges.dc, mesh.edges.cellsOnEdge)

function gradient_at_edge!(out::AbstractVector, c_field, dc, cellsOnEdge, op::F = Base.identity) where {F <: Function}
    @parallel for e in eachindex(cellsOnEdge)
        @inbounds @inline begin
            c1, c2 = cellsOnEdge[e]
            out[e] = (op(c_field[c2]) - op(c_field[c1])) / dc[e]
        end
    end
    return out
end

function gradient_at_edge!(out::AbstractVector, op_out::F, c_field, dc, cellsOnEdge, op::F2 = Base.identity) where {F <: Function, F2 <: Function}
    @parallel for e in eachindex(cellsOnEdge)
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

    @parallel for e in eachindex(cellsOnEdge)
        @inbounds @inline begin
            c1, c2 = cellsOnEdge[e]
            inv_dc = inv(dc[e])
            inv_dc_simd = simd_repeat(ValN_SIMD, inv_dc)

            for k in range_simd
                k_simd = lane + k
                out[k_simd, e] = inv_dc_simd * (op(c_field[k_simd, c2]) - op(c_field[k_simd, c1]))
            end

            for k in range_serial
                out[k, e] = inv_dc * (op(c_field[k, c2]) - op(c_field[k, c1]))
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

    @parallel for e in eachindex(cellsOnEdge)
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

    @parallel for e in eachindex(cellsOnEdge)
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

    @parallel for e in eachindex(cellsOnEdge)
        @inbounds @inline begin
            c1, c2 = cellsOnEdge[e]
            inv_dc = inv(dc[e])
            inv_dc_simd = simd_repeat(ValN_SIMD, inv_dc)

            for t in axes(out, 3)

                for k in range_simd
                    k_simd = lane + k
                    out[k_simd, e, t] = inv_dc_simd * (op(c_field[k_simd, c2, t]) - op(c_field[k_simd, c1, t]))
                end

                for k in range_serial
                    out[k, e, t] = inv_dc * (op(c_field[k, c2, t]) - op(c_field[k, c1, t]))
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

    @parallel for e in eachindex(cellsOnEdge)
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

    @parallel for e in eachindex(cellsOnEdge)
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

function (Vop::GradientAtEdge)(in_field::AbstractArray, op::F = Base.identity) where {F <: Function}
    is_proper_size(in_field, n_input(Vop)) || throw(DimensionMismatch("Input array doesn't seem to be a $(name_input(Vop)) field"))
    s = construct_new_node_index(size(in_field)..., n_output(Vop))
    out_field = my_similar(in_field, Base.promote_op(*, eltype(eltype(Vop.dc)), Base.promote_op(op, eltype(in_field))), s)
    return Vop(out_field, in_field, op)
end

struct DivAtCell{N_MAX, TI, TF} <: DifferentialOperator
    n::Int
    indices::Vector{ImmutableVector{N_MAX, TI}}
    weights::Vector{ImmutableVector{N_MAX, TF}}
end

name_input(::DivAtCell) = "edge"
name_output(::DivAtCell) = "cell"

function compute_div_at_cell_weights(areaCell, edgesOnCell::Vector{<:ImmutableVector{N_MAX}}, dvEdge::AbstractVector{T}, cellsOnEdge) where {N_MAX, T}
    w = Vector{ImmutableVector{N_MAX, T}}(undef, length(areaCell))
    aux = Vector{T}(undef, N_MAX)

    @inbounds for c in eachindex(areaCell)
        inv_a = inv(areaCell[c])

        fill!(aux, zero(T))
        eoc = edgesOnCell[c]
        l = length(eoc)
        for i in Base.OneTo(l)
            e = eoc[i]
            Le = dvEdge[e]
            aux[i] = Le * inv_a * VoronoiMeshDataStruct.sign_edge(cellsOnEdge[e], c)
        end
        w[c] = ImmutableVector{N_MAX}(ntuple(j -> getindex(aux, j), Val{N_MAX}()), l)
    end
    return w
end

function DivAtCell(mesh::VoronoiMesh)
    w = compute_div_at_cell_weights(mesh.areaCell, mesh.edgesOnCell, mesh.dvEdge, mesh.cellsOnEdge)
    return DivAtCell(mesh.edges.n, mesh.edgesOnCell, w)
end

struct CurlAtVertex{TF, TI} <: DifferentialOperator
    n::Int
    weights::Vector{NTuple{3, TF}}
    indices::Vector{NTuple{3, TI}}
end

name_input(::CurlAtVertex) = "edge"
name_output(::CurlAtVertex) = "vertex"

function compute_rotational_at_vertex_weights!(weights, areaVertex, dc, edgesOnVertex, cellsOnVertex, cellsOnEdge)

    @parallel for v in eachindex(weights)
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
    weights = similar(areaVertex, NTuple{3, Base.promote_op(/, eltype(dc), eltype(areaVertex))})
    return compute_rotational_at_vertex_weights!(weights, areaVertex, dc, edgesOnVertex, cellsOnVertex, cellsOnEdge)
end

CurlAtVertex(mesh::VoronoiMesh) = CurlAtVertex(mesh.edges.n, compute_rotational_at_vertex_weights(mesh.vertices.area, mesh.edges.dc, mesh.edgesOnVertex, mesh.cellsOnVertex, mesh.cellsOnEdge), mesh.edgesOnVertex)
