module NCDatasetsExt

using NCDatasets
using TensorsLite, SmallCollections, VoronoiMeshes, VoronoiOperators
import VoronoiOperators: VoronoiOperator, n_output, method_name
import VoronoiMeshes: save_to_netcdf, save_to_netcdf!

function save_to_netcdf(filename::String, obj::VoronoiOperator; format = :netcdf5_64bit_data)

    mode = isfile(filename) ? "a" : "c"

    NCDataset(filename, mode, format = format) do ds
        save_to_netcdf!(ds, obj)
    end
end

function save_to_netcdf!(ds::NCDataset, tanVelRecon::TangentialVelocityReconstruction{N, TI, TF}) where {N, TI, TF}

    write_field_to_netcdf!(ds, tanVelRecon.weights, "coeffs_edge_tanVel_reconstruct", ("trueMaxEdges2", "nEdges"), ["units" => "-", "long_name" => "Weights used in reconstruction of tangential velocity for an edge."])

    ds.attrib["tangential_velocity_reconstruction_method"] = method_name(tanVelRecon)
    return ds
end

function save_to_netcdf!(ds::NCDataset, velRecon::CellVelocityReconstruction{N_MAX, TI, TF, TZ}) where {N_MAX, TI, TF, TZ}

    write_field_to_netcdf!(ds, velRecon.weights, "coeffs_cell_vel_reconstruct", ("R3", "maxEdges", "nCells"), ["units" => "-", "long_name" => "Coefficients to reconstruct velocity vectors at cell centers"])

    ds.attrib["cell_velocity_reconstruction_method"] = method_name(velRecon)

    return
end

function save_to_netcdf!(ds::NCDataset, velRecon::VertexVelocityReconstruction{TI, TF, TZ}) where {TI, TF, TZ}

    write_field_to_netcdf!(ds, velRecon.weights, "coeffs_vertex_vel_reconstruct", ("R3", "vertexDegree", "nVertices"), ["units" => "-", "long_name" => "Coefficients to reconstruct velocity vectors at vertices"])

    ds.attrib["vertex_velocity_reconstruction_method"] = method_name(velRecon)

    return
end

end
