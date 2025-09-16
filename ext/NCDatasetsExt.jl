module NCDatasetsExt

using TensorsLite, SmallCollections, VoronoiMeshes, VoronoiOperators
import VoronoiOperators: VoronoiOperator, n_output, method_name
import VoronoiMeshes: save_to_netcdf, save_to_netcdf!
using NCDatasets

function save_to_netcdf(filename::AbstractString, obj::VoronoiOperator; format = :netcdf5_64bit_data)

    mode = isfile(filename) ? "a" : "c"

    NCDataset(filename, mode, format = format) do ds
        save_to_netcdf!(ds, obj)
    end
end

function save_to_netcdf!(ds::NCDataset, tanVelRecon::TangentialVelocityReconstruction{N, TI, TF}) where {N, TI, TF}

    if !haskey(ds.dim, "maxEdges2")
        ds.dim["maxEdges2"] = N
    end

    if !haskey(ds.dim, "nEdges")
        ds.dim["nEdges"] = n_output(tanVelRecon)
    end

    defVar(
        ds, "weightsOnEdge", reinterpret(reshape, TF, tanVelRecon.weights.data),
        ("maxEdges2", "nEdges"), attrib = [
            "units" => "-",
            "long_name" => "Weights used in reconstruction of tangential velocity for an edge.",
        ]
    )

    ds.attrib["tangential_velocity_reconstruction_method"] = method_name(tanVelRecon)
    return ds
end

function save_to_netcdf!(ds::NCDataset, velRecon::CellVelocityReconstruction{N_MAX, TI, TF, TZ}) where {N_MAX, TI, TF, TZ}

    ncells = length(velRecon.weights)

    if !haskey(ds.dim, "R3")
        ds.dim["R3"] = 3
    end

    if !haskey(ds.dim, "maxEdges")
        ds.dim["maxEdges"] = N_MAX
    end

    if !haskey(ds.dim, "nCells")
        ds.dim["nCells"] = ncells
    end

    weights = if (TZ === TF)
        reshape(reinterpret(TF, velRecon.weights.data), (3, N_MAX, ncells))
    else
        wdata = velRecon.weights.data
        w = Array{TF}(undef, (3, N_MAX, ncells))
        @inbounds for k in Base.OneTo(ncells)
            t = wdata[k]
            for j in Base.OneTo(N_MAX)
                v = t[j]
                w[1, j, k] = v.x
                w[2, j, k] = v.y
                w[3, j, k] = zero(TF)
            end
        end
        w
    end

    defVar(
        ds, "coeffs_reconstruct", weights,
        ("R3", "maxEdges", "nCells"), attrib = [
            "units" => "-",
            "long_name" => "Coefficients to reconstruct velocity vectors at cell centers",
        ]
    )

    ds.attrib["cell_velocity_reconstruction_method"] = method_name(velRecon)

    return
end

function save_to_netcdf!(ds::NCDataset, velRecon::VertexVelocityReconstruction{TI, TF, TZ}) where {TI, TF, TZ}

    nvertices = length(velRecon.weights)

    if !haskey(ds.dim, "R3")
        ds.dim["R3"] = 3
    end

    if !haskey(ds.dim, "vertexDegree")
        ds.dim["vertexDegree"] = 3
    end

    if !haskey(ds.dim, "nVertices")
        ds.dim["nVertices"] = nvertices
    end

    weights = if (TZ === TF)
        reshape(reinterpret(TF, velRecon.weights), (3, 3, nvertices))
    else
        wdata = velRecon.weights
        w = Array{TF}(undef, (3, 3, nvertices))
        @inbounds for k in Base.OneTo(nvertices)
            t = wdata[k]
            for j in Base.OneTo(3)
                v = t[j]
                w[1, j, k] = v.x
                w[2, j, k] = v.y
                w[3, j, k] = zero(TF)
            end
        end
        w
    end

    defVar(
        ds, "vertex_coeffs_reconstruct", weights,
        ("R3", "vertexDegree", "nVertices"), attrib = [
            "units" => "-",
            "long_name" => "Coefficients to reconstruct velocity vectors at vertices",
        ]
    )

    ds.attrib["vertex_velocity_reconstruction_method"] = method_name(velRecon)

    return
end

end
