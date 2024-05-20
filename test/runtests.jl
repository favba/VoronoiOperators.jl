using TensorsLite, TensorsLiteGeometry
using NCDatasets
using VoronoiMeshDataStruct
using VoronoiOperators
using Test

const mesh_iso = NCDataset("mesh.nc") do f; VoronoiMesh(f) end
const mesh_distorted = NCDataset("mesh_distorted.nc") do f; VoronoiMesh(f) end
const ncells = mesh_iso.cells.n
const nvertex = mesh_iso.vertices.n
const nedges = mesh_iso.edges.n

@testset "Vertex to Edge Transformations" begin

    vert_const_field1D = ones(nvertex)
    vert_const_field2D = ones(8,nvertex)
    vert_const_field3D = ones(8,nvertex,2)

    vert_const_vec_field1D = VecArray(x=ones(nvertex),y=ones(nvertex))
    vert_const_vec_field2D = VecArray(x=ones(8,nvertex),y=ones(8,nvertex))
    vert_const_vec_field3D = VecArray(x=ones(8,nvertex,2),y=ones(8,nvertex,2))

    for mesh in (mesh_iso,mesh_distorted)
        for v2e in (VertexToEdgeMean(mesh), VertexToEdgeInterpolation(mesh), VertexToEdgePiecewise(mesh), VertexToEdgeArea(mesh))
            for field in (vert_const_field1D,vert_const_field2D,vert_const_field3D)
                @test all(isequal(1),v2e(field))
                e_field = v2e(field)
                @test all(isequal(2),v2e(e_field,+,field))
            end
            for field in (vert_const_vec_field1D,vert_const_vec_field2D,vert_const_vec_field3D)
                @test all(isequal(1.0𝐢+1.0𝐣),v2e(field))
                e_field = v2e(field)
                @test all(isequal(2.0𝐢+2.0𝐣),v2e(e_field,+,field))
            end
        end
    end

end

@testset "Cell to Edge Transformations" begin
    cell_const_field1D = ones(ncells)
    cell_const_field2D = ones(8,ncells)
    cell_const_field3D = ones(8,ncells,2)

    cell_const_vec_field1D = VecArray(x=ones(ncells),y=ones(ncells))
    cell_const_vec_field2D = VecArray(x=ones(8,ncells),y=ones(8,ncells))
    cell_const_vec_field3D = VecArray(x=ones(8,ncells,2),y=ones(8,ncells,2))

    for mesh in (mesh_iso,mesh_distorted)
        for c2e in (CellToEdgeMean(mesh), CellToEdgeBaricentric(mesh))
            for field in (cell_const_field1D,cell_const_field2D,cell_const_field3D)
                @test all(isapprox(1),c2e(field))
                e_field = c2e(field)
                @test all(isapprox(2),c2e(e_field,+,field))
            end
            for field in (cell_const_vec_field1D,cell_const_vec_field2D,cell_const_vec_field3D)
                @test all(isapprox(1.0𝐢+1.0𝐣),c2e(field))
                e_field = c2e(field)
                @test all(isapprox(2.0𝐢+2.0𝐣),c2e(e_field,+,field))
            end
        end
    end

end

@testset "Kinetic Energy Reconstruction" begin
    edge_const_field1D = ones(nedges)
    edge_const_field2D = ones(8,nedges)
    edge_const_field3D = ones(8,nedges,2)
    for mesh in (mesh_iso,mesh_distorted)
        for kc in (CellKineticEnergyRingler(mesh), CellKineticEnergyMPAS(mesh))
            for field in (edge_const_field1D,edge_const_field2D,edge_const_field3D)
                @test all(>(0),kc(field))
                s = ndims(field) == 1 ? (ncells,) : ndims(field) == 2 ? (8,ncells) : (8,ncells,2)
                f = ones(s)
                @test kc(f,+,field) ≈ (kc(field) .+ 1)
            end
        end
    end
end