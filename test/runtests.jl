using TensorsLite, TensorsLiteGeometry, ImmutableVectors
using NCDatasets
using VoronoiMeshDataStruct
using VoronoiOperators
using Test

const mesh_iso = NCDataset("mesh.nc") do f
    VoronoiMesh(f)
end
const mesh_distorted = NCDataset("mesh_distorted.nc") do f
    VoronoiMesh(f)
end
const ncells = mesh_iso.cells.n
const nvertex = mesh_iso.vertices.n
const nedges = mesh_iso.edges.n

@testset "Vertex to Edge Transformations" begin

    vert_const_field1D = ones(nvertex)
    vert_const_field2D = ones(8, nvertex)
    vert_const_field3D = ones(8, nvertex, 2)

    vert_const_vec_field1D = VecArray(x=ones(nvertex), y=ones(nvertex))
    vert_const_vec_field2D = VecArray(x=ones(8, nvertex), y=ones(8, nvertex))
    vert_const_vec_field3D = VecArray(x=ones(8, nvertex, 2), y=ones(8, nvertex, 2))

    for mesh in (mesh_iso, mesh_distorted)
        for v2e in (VertexToEdgeMean(mesh), VertexToEdgeInterpolation(mesh), VertexToEdgePiecewise(mesh), VertexToEdgeArea(mesh))
            for field in (vert_const_field1D, vert_const_field2D, vert_const_field3D)
                @test all(isequal(1), v2e(field))
                e_field = v2e(field)
                @test all(isequal(2), v2e(e_field, +, field))
            end
            for field in (vert_const_vec_field1D, vert_const_vec_field2D, vert_const_vec_field3D)
                @test all(isequal(1.0𝐢 + 1.0𝐣), v2e(field))
                e_field = v2e(field)
                @test all(isequal(2.0𝐢 + 2.0𝐣), v2e(e_field, +, field))
            end
        end
    end

end

@testset "Cell to Edge Transformations" begin
    cell_const_field1D = ones(ncells)
    cell_const_field2D = ones(8, ncells)
    cell_const_field3D = ones(8, ncells, 2)

    cell_const_vec_field1D = VecArray(x=ones(ncells), y=ones(ncells))
    cell_const_vec_field2D = VecArray(x=ones(8, ncells), y=ones(8, ncells))
    cell_const_vec_field3D = VecArray(x=ones(8, ncells, 2), y=ones(8, ncells, 2))

    for mesh in (mesh_iso, mesh_distorted)
        for c2e in (CellToEdgeMean(mesh), CellToEdgeBaricentric(mesh))
            for field in (cell_const_field1D, cell_const_field2D, cell_const_field3D)
                @test all(isapprox(1), c2e(field))
                e_field = c2e(field)
                @test all(isapprox(2), c2e(e_field, +, field))
            end
            for field in (cell_const_vec_field1D, cell_const_vec_field2D, cell_const_vec_field3D)
                @test all(isapprox(1.0𝐢 + 1.0𝐣), c2e(field))
                e_field = c2e(field)
                @test all(isapprox(2.0𝐢 + 2.0𝐣), c2e(e_field, +, field))
            end
        end
    end

end

const v = 3.0𝐢 + 4.0𝐣

@testset "Cell Velocity / Kinetic Energy Reconstruction" begin
    for mesh in (mesh_iso, mesh_distorted)
        isdefined(mesh.edges, :normalVectors) || compute_edge_normals!(mesh)
        ue1D = dot.(mesh.edges.normalVectors, (v,))
        ue2D = similar(ue1D, (8, nedges))
        ue3D = similar(ue1D, (8, nedges, 2))
        for k in 1:8
            ue2D[k, :] .= ue1D
        end
        for t in 1:2
            ue3D[:, :, t] .= ue2D
        end
        for uR in (CellVelocityReconstructionPerot(mesh),)
            for ueND in (ue1D, ue2D, ue3D)
                @test all(isapprox(3.0𝐢 + 4.0𝐣), uR(ueND))
                field = uR(ueND)
                @test all(isapprox(6.0𝐢 + 8.0𝐣), uR(field, +, ueND))

                for kR in (CellKineticEnergyVelRecon(uR),)
                    @test all(isapprox(12.5), kR(ueND))
                    fieldk = kR(ueND)
                    @test all(isapprox(25.0), kR(fieldk, +, ueND))
                end
            end
        end
        for kR in (CellKineticEnergyRingler(mesh), CellKineticEnergyMPAS(mesh))
            for ueND in (ue1D, ue2D, ue3D)
                if (mesh === mesh_iso)
                    @test all(isapprox(12.5), kR(ueND))
                    fieldk = kR(ueND)
                    @test all(isapprox(25.0), kR(fieldk, +, ueND))
                else
                    @test all(x -> isapprox(x, 12.5; atol=3.0), kR(ueND))
                    fieldk = kR(ueND)
                    @test all(x -> isapprox(x, 25.0; atol=6.0), kR(fieldk, +, ueND))
                end
            end
        end
    end
end

@testset "Gradient at Edge" begin
    for mesh in (mesh_iso, mesh_distorted)
        c_field = dot.((v,), mesh.cells.position)
        grad_c_field = dot.((v,), mesh.edges.normalVectors)
        mask = periodic_edges_mask(mesh)
        grad_c_masked = grad_c_field[mask]

        ∇e = GradientAtEdge(mesh)
        @test all(x -> isapprox(x[1], x[2]), zip(grad_c_masked, ∇e(c_field)[mask]))

        e_field = ∇e(c_field)
        @test all(x -> isapprox(2 * x[1], x[2]), zip(grad_c_masked, ∇e(e_field, +, c_field)[mask]))
        @test all(x -> isapprox(x[1], x[2]), zip(grad_c_masked, ∇e(e_field, -, c_field)[mask]))
    end
end

@testset "Divergence at Cell" begin
    for mesh in (mesh_iso, mesh_distorted)
        isdefined(mesh.edges, :normalVectors) || compute_edge_normals!(mesh)
        ue1D = dot.(mesh.edges.normalVectors, (v,))
        ue2D = similar(ue1D, (8, nedges))
        ue3D = similar(ue1D, (8, nedges, 2))
        for k in 1:8
            ue2D[k, :] .= ue1D
        end
        for t in 1:2
            ue3D[:, :, t] .= ue2D
        end
        Div = DivAtCell(mesh)
        for ueND in (ue1D, ue2D, ue3D)
            atol = 1e-8 * norm(v) # isapprox(0.0) is tricky to evaluate
            @test all(x -> isapprox(x, 0.0; atol=atol), Div(ueND))
            field = Div(ueND)
            field .= 1
            @test all(x -> isapprox(x, 1.0), Div(field, +, ueND))
        end
    end
end

@testset "Cell value Filtering" begin
    cell_const_field1D = ones(ncells)
    cell_const_field2D = ones(8, ncells)
    cell_const_field3D = ones(8, ncells, 2)

    cell_const_vec_field1D = VecArray(x=ones(ncells), y=ones(ncells))
    cell_const_vec_field2D = VecArray(x=ones(8, ncells), y=ones(8, ncells))
    cell_const_vec_field3D = VecArray(x=ones(8, ncells, 2), y=ones(8, ncells, 2))

    for mesh in (mesh_iso, mesh_distorted)
        Filter = CellBoxFilter(mesh)

        for field in (cell_const_field1D, cell_const_field2D, cell_const_field3D)
            @test all(isapprox(1), Filter(field))
            e_field = Filter(field)
            @test all(isapprox(2), Filter(e_field, +, field))
        end

        for field in (cell_const_vec_field1D, cell_const_vec_field2D, cell_const_vec_field3D)
            @test all(isapprox(1.0𝐢 + 1.0𝐣), Filter(field))
            e_field = Filter(field)
            @test all(isapprox(2.0𝐢 + 2.0𝐣), Filter(e_field, +, field))
        end

        VariableFilter = CellBoxFilter(mesh, true)

        for field in (cell_const_field1D, cell_const_field2D, cell_const_field3D)
            @test all(isapprox(1), VariableFilter(field))
            e_field = VariableFilter(field)
            @test all(isapprox(2), VariableFilter(e_field, +, field))
        end

        for field in (cell_const_vec_field1D, cell_const_vec_field2D, cell_const_vec_field3D)
            @test all(isapprox(1.0𝐢 + 1.0𝐣), VariableFilter(field))
            e_field = VariableFilter(field)
            @test all(isapprox(2.0𝐢 + 2.0𝐣), VariableFilter(e_field, +, field))
        end
    end
end
