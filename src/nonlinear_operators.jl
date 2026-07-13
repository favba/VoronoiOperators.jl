abstract type AbstractDivCellScalarU <: NonLinearVoronoiOperator end
name_input(::AbstractDivCellScalarU) = "cell, edge"
name_output(::AbstractDivCellScalarU) = "cell"

struct DivCellScalarU{TI, TF, CTE<:CellToEdgeTransformation, DIV<:CellDiv{TI, TF}} <: AbstractDivCellScalarU
    cellToEdge::CTE
    div::DIV
end

DivCellScalarU(mesh::AbstractVoronoiMesh) = DivCellScalarU(CellToEdgeMean(mesh), DivAtCell(mesh))

function (divrhou::DivCellScalarU)(out::AbstractArray, u_inter::AbstractArray, ρ::AbstractArray, u::AbstractArray)
    divrhou.cellToEdge(u_inter, ρ)
    mytmap!(*, u_inter, u, u_inter)
    divrhou.div(out, u_inter)
end

function (divrhou::DivCellScalarU)(out::AbstractArray, ρ::AbstractArray, u::AbstractArray)
    u_iter = create_output_array(divrhou.cellToEdge, ρ)
    return divrhou(out, u_iter, ρ, u)
end

function (divrhou::DivCellScalarU)(ρ::AbstractArray, u::AbstractArray)
    out = create_output_array(divrhou.div, u)
    return divrhou(out, ρ, u)
end


#This computes -pv*h*u⟂, where pv = (∇⨯uₕ + f) / h, following the original TRiSK scheme
@inline function local_trisk_coriolis_term(Inds::TT, we::AbstractVector{TF}, eoe::AbstractVector{<:Integer}, h_edge, u, pv_edge) where {TT<:Tuple, TF}

    @inbounds begin
        pve_half = pv_edge[Inds...] / 2
        r = zero(TF)

        for i in eachindex(eoe)
            ep = Int(eoe[i])
            Indsp = construct_new_node_index(ep, Inds...)
            r = muladd(we[i], h_edge[Indsp...] * u[Indsp...] * muladd(0.5, pv_edge[Indsp...], pve_half), r)
        end
    end

    return -r
end

#This computes -pv*h*u⟂, where pv = (∇⨯uₕ + f) / h, following the original TRiSK scheme
function trisk_coriolis_term!(output::AbstractVector{TF}, weightsOnEdge::AbstractVector, edgesOnEdge, h_edge, u, pv_edge) where {TF}
    @batch for e in eachindex(output)
        @inbounds output[e] = local_trisk_coriolis_term((e,), weightsOnEdge[e], edgesOnEdge[e], h_edge, u, pv_edge)
    end
    return output
end


#This computes -pv*h*u⟂, where pv = (∇⨯uₕ + f) / h, following the original TRiSK scheme
function trisk_coriolis_term!(output::AbstractMatrix{TF}, weightsOnEdge::AbstractVector, edgesOnEdge, h_edge, u, pv_edge) where {TF}

    N_SIMD = simd_length(TF)
    ValN_SIMD = Val{N_SIMD}()
    lane = sd.VecRange{N_SIMD}(0)
    Nk = size(output, 1)

    range_simd, range_serial = simd_ranges(N_SIMD, Nk)
    is_there_rest = length(range_serial) != 0
    k_simd_end = lane + (Nk - N_SIMD + 1)

    @batch for e in axes(output, 2)
        @inbounds begin
            eoe = map(Int, edgesOnEdge[e])
            we = weightsOnEdge[e]
            we_simd = simd_repeat(ValN_SIMD, we)

            for k in range_simd
                k_simd = lane + k

                output[k_simd, e] = local_trisk_coriolis_term(
                    (k_simd, e), we_simd, eoe, h_edge, u, pv_edge
                )
            end

            if is_there_rest
                output[k_simd_end, e] = local_trisk_coriolis_term(
                    (k_simd_end, e), we_simd, eoe, h_edge, u, pv_edge
                )
            end
        end #inbounds
    end

    return output
end

trisk_coriolis_term!(output::AbstractArray{N},
    tR::TangentialVelocityReconstruction,
    h_edge::AbstractArray{N}, u::AbstractArray{N},
    pv_edge::AbstractArray{N}) where {N} =
        trisk_coriolis_term!(output, tR.weights, tR.indices, h_edge, u, pv_edge)


#This computes -(∇×uₕ + f)*u⟂ in the same manner as is done on MPAS 
@inline function local_mpas_coriolis_term(Inds::TT, we::AbstractVector{TF}, eoe::AbstractVector{<:Integer}, u, absolute_vorticity_edge) where {TT<:Tuple, TF}

    @inbounds begin
        vort_half = absolute_vorticity_edge[Inds...] / 2
        r = zero(TF)

        for i in eachindex(eoe)
            ep = Int(eoe[i])
            Indsp = construct_new_node_index(ep, Inds...)
            r = muladd(we[i],  u[Indsp...] * muladd(0.5, absolute_vorticity_edge[Indsp...], vort_half), r)
        end
    end

    return -r
end

#This computes -(∇×uₕ + f)*u⟂ in the same manner as is done on MPAS 
function mpas_coriolis_term!(output::AbstractVector{TF}, weightsOnEdge::AbstractVector, edgesOnEdge::AbstractVector, u::AbstractVector, absolute_vorticity_edge::AbstractVector) where {TF}
    @batch for e in eachindex(output)
        @inbounds begin
            eoe = edgesOnEdge[e]
            we = weightsOnEdge[e]

            output[e] = local_mpas_coriolis_term(
                (e,), we, eoe, u, absolute_vorticity_edge
            )
        end
    end
    return output
end

#This computes -(∇×uₕ + f)*u⟂ in the same manner as is done on MPAS 
function mpas_coriolis_term!(output::AbstractMatrix{TF}, weightsOnEdge::AbstractVector, edgesOnEdge::AbstractVector, u::AbstractMatrix, absolute_vorticity_edge::AbstractMatrix) where {TF}

    N_SIMD = simd_length(TF)
    ValN_SIMD = Val{N_SIMD}()
    lane = sd.VecRange{N_SIMD}(0)
    Nk = size(output, 1)

    range_simd, range_serial = simd_ranges(N_SIMD, Nk)
    is_there_rest = length(range_serial) != 0
    k_simd_end = lane + (Nk - N_SIMD + 1)

    @batch for e in axes(output, 2)
        @inbounds begin
            eoe = map(Int, edgesOnEdge[e])
            we = weightsOnEdge[e]
            we_simd = simd_repeat(ValN_SIMD, we)

            for k in range_simd
                k_simd = lane + k

                output[k_simd, e] = local_mpas_coriolis_term(
                    (k_simd, e), we_simd, eoe, u, absolute_vorticity_edge
                )
            end

            if is_there_rest
                output[k_simd_end, e] = local_mpas_coriolis_term(
                    (k_simd_end, e), we_simd, eoe, u, absolute_vorticity_edge
                )
            end
        end #inbounds
    end

    return output
end

mpas_coriolis_term!(output::AbstractArray{N},
    tR::TangentialVelocityReconstruction,
    u::AbstractArray{N}, absolute_vorticity_edge::AbstractArray{N}) where {N} =
        mpas_coriolis_term!(output, tR.weights, tR.indices, u, absolute_vorticity_edge)

