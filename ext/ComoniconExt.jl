module ComoniconExt

using NCDatasets

using VoronoiMeshes, VoronoiOperators
const vm = Base.get_extension(VoronoiMeshes, :NCDatasetsExt)

using VoronoiOperators
const vo = Base.get_extension(VoronoiOperators, :NCDatasetsExt)

using Comonicon

"""
Create and save new Voronoi horizontal operators to a NetCDF file.

# Arguments

- `grid`: NetCDF file containing Voronoi grid information
- `output`: NetCDF file name for output

# Options

- `-t, --tangent_reconstruction`: Edge tangential velocity reconstruction method. Available options: "Peixoto", "PeixotoOld", "LSq1", "LSq2", and "Thuburn". Default is "Thuburn".
- `-c, --cell_reconstruction`: Cell velocity reconstruction method. Available options: "Perot", "PerotOld", "LSq1", and "LSq2". No cell velocity reconstruction method is written to file by default.
"""
Comonicon.@main function create_voronoi_operator(grid::String, output::String; tangent_reconstruction::String="Thuburn", cell_reconstruction::String="")
    @info string("Reading ", grid," file")
    mesh = vm.VoronoiMesh(grid)
    @info "Creating edge tangent velocity reconstruction weights"
    VoronoiOperators.save_tangent_reconstruction(mesh, tangent_reconstruction, output)
    if cell_reconstruction != ""
        @info "Creating cell velocity reconstruction weights"
        VoronoiOperators.save_cell_reconstruction(mesh, cell_reconstruction, output)
    end
    @info string("Operators saved to ", output)
    return 0
end

precompile(Tuple{typeof(Core.kwcall), NamedTuple{(:tangent_reconstruction, :cell_reconstruction), Tuple{String, String}}, typeof(create_voronoi_operator), String, String})
precompile(create_voronoi_operator, (String, String, @NamedTuple{tangent_reconstruction::String, cell_reconstruction::String}))

precompile(Tuple{typeof(command_main), Vector{String}})
precompile(Tuple{typeof(command_main)})

end
