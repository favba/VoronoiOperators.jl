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
