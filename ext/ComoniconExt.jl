module ComoniconExt

using NCDatasets

using VoronoiMeshes, VoronoiOperators

using VoronoiOperators

using Comonicon

function save_start_time(fn::String, st::String)
    NCDataset(fn,"a") do ds

        if !haskey(ds.dim, "StrLen")
            ds.dim["StrLen"] = 64
        end

        if !haskey(ds.dim, "Time")
            ds.dim["Time"] = Inf #Unlimited dimension
        end

        if !haskey(ds, "xtime")
            xtime = [' ' for i=1:64, j=1:1]
            nl = length(st)
            xtime[1:nl,1] .= collect(st)
            defVar(
                ds, "xtime", xtime,
                ("StrLen", "Time"), attrib = [
                    "units" => "YYYY-MM-DD_hh:mm:ss",
                    "long_name" => "Model valid time",
                ]
            )
        end
    end
end

precompile(Tuple{typeof(save_start_time), String, String})

"""
Create and save new Voronoi horizontal operators to a NetCDF file.

# Arguments

- `grid`: NetCDF file containing Voronoi grid information
- `output`: NetCDF file name for output

# Options

- `-t, --tangent_reconstruction`: Edge tangential velocity reconstruction method. Available options: "Peixoto", "PeixotoOld", "LSq1", "LSq2", and "Thuburn". Default is "Thuburn".
- `-c, --cell_reconstruction`: Cell velocity reconstruction method. Available options: "Perot", "PerotOld", "LSq1", and "LSq2". No cell velocity reconstruction method is written to file by default.
- `-v, --vertex_reconstruction`: Vertex velocity reconstruction method. Available options: "Perot", "LSq1", and "LSq2". No vertex velocity reconstruction method is written to file by default.
- `-s, --start_time`: Simulation start time. Should be the same as in "config \U0332 start \U0332 time" from namelist.init \U0332 atmosphere. Default value of "0000-01-01 \U0332 00:00:00".
"""
Comonicon.@main function create_voronoi_operator(grid::String, output::String;
                                                 tangent_reconstruction::String="Thuburn",
                                                 cell_reconstruction::String="",
                                                 vertex_reconstruction::String="",
                                                 start_time::String="0000-01-01_00:00:00")
    @info string("Reading ", grid," file")
    mesh = VoronoiMesh(grid, read_computed=false)
    @info "Creating edge tangent velocity reconstruction weights"
    VoronoiOperators.save_tangent_reconstruction(mesh, tangent_reconstruction, output)
    if cell_reconstruction != ""
        @info "Creating cell velocity reconstruction weights"
        VoronoiOperators.save_cell_reconstruction(mesh, cell_reconstruction, output)
    end
    if vertex_reconstruction != ""
        @info "Creating vertex velocity reconstruction weights"
        VoronoiOperators.save_vertex_reconstruction(mesh, vertex_reconstruction, output)
    end
    save_start_time(output, start_time)
    @info string("Operators saved to ", output)
    return 0
end

precompile(Tuple{typeof(Core.kwcall), NamedTuple{(:tangent_reconstruction, :cell_reconstruction, :vertex_reconstruction, :start_time), Tuple{String, String, String, String}}, typeof(create_voronoi_operator), String, String})
precompile(create_voronoi_operator, (String, String, @NamedTuple{tangent_reconstruction::String, cell_reconstruction::String, vertex_reconstruction::String, start_time::String}))

precompile(Tuple{typeof(command_main), Vector{String}})
precompile(Tuple{typeof(command_main)})

VoronoiOperators.create_voronoi_operator(args::Vector{String}) = command_main(args)

precompile(Tuple{typeof(VoronoiOperators.create_voronoi_operator), Vector{String}})
end
