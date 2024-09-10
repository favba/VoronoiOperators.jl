abstract type DifferentialOperator <: LinearVoronoiOperator end

struct GradientAtEdge{TI,TF} <: DifferentialOperator
    nCells::Int
    dc::Vector{TF}
    cellsOnEdge::Vector{NTuple{2,TI}}
end

GradientAtEdge(mesh::VoronoiMesh) = GradientAtEdge(mesh.cells.n,mesh.edges.dc,mesh.edges.cellsOnEdge)

function gradient_at_edge!(out::AbstractVector,c_field,dc,cellsOnEdge)
    @parallel for e in eachindex(cellsOnEdge)
        @inbounds begin
        c1,c2 = cellsOnEdge[e]
        out[e] = (c_field[c2] - c_field[c1])/dc[e]
        end
    end
    return out
end

function gradient_at_edge!(out::AbstractVector,op::F,c_field,dc,cellsOnEdge) where F<:Function
    @parallel for e in eachindex(cellsOnEdge)
        @inbounds begin
        c1,c2 = cellsOnEdge[e]
        out[e] = op(out[e],(c_field[c2] - c_field[c1])/dc[e])
        end
    end
    return out
end

function gradient_at_edge!(out::AbstractMatrix,c_field,dc,cellsOnEdge)
    @parallel for e in eachindex(cellsOnEdge)
        @inbounds begin
        c1,c2 = cellsOnEdge[e]
        inv_dc = inv(dc[e])
        @simd ivdep for k in axes(out,1)
            out[k,e] = inv_dc*(c_field[k,c2] - c_field[k,c1])
        end
        end #inbounds
    end
    return out
end

function gradient_at_edge!(out::AbstractMatrix,op::F,c_field,dc,cellsOnEdge) where F<:Function
    @parallel for e in eachindex(cellsOnEdge)
        @inbounds begin
        c1,c2 = cellsOnEdge[e]
        inv_dc = inv(dc[e])
        @simd ivdep for k in axes(out,1)
            out[k,e] = op(out[k,e],inv_dc*(c_field[k,c2] - c_field[k,c1]))
        end
        end #inbounds
    end
    return out
end

function gradient_at_edge!(out::AbstractMatrix,op::typeof(+),c_field,dc,cellsOnEdge)
    @parallel for e in eachindex(cellsOnEdge)
        @inbounds begin
        c1,c2 = cellsOnEdge[e]
        inv_dc = inv(dc[e])
        @simd ivdep for k in axes(out,1)
            out[k,e] = muladd(inv_dc,(c_field[k,c2] - c_field[k,c1]),out[k,e])
        end
        end #inbounds
    end
    return out
end

function gradient_at_edge!(out::AbstractArray{<:Any,3},c_field,dc,cellsOnEdge)
    @parallel for e in eachindex(cellsOnEdge)
        @inbounds begin
        c1,c2 = cellsOnEdge[e]
        inv_dc = inv(dc[e])
        for t in axes(out,3)
            @simd ivdep for k in axes(out,1)
                out[k,e,t] = inv_dc*(c_field[k,c2,t] - c_field[k,c1,t])
            end
        end
        end #inbounds
    end
    return out
end

function gradient_at_edge!(out::AbstractArray{<:Any,3},op::F,c_field,dc,cellsOnEdge) where F<:Function
    @parallel for e in eachindex(cellsOnEdge)
        @inbounds begin
        c1,c2 = cellsOnEdge[e]
        inv_dc = inv(dc[e])
        for t in axes(out,3)
            @simd ivdep for k in axes(out,1)
                out[k,e,t] = op(out[k,e,t],inv_dc*(c_field[k,c2,t] - c_field[k,c1,t]))
            end
        end
        end #inbounds
    end
    return out
end

function gradient_at_edge!(out::AbstractArray{<:Any,3},op::typeof(+),c_field,dc,cellsOnEdge)
    @parallel for e in eachindex(cellsOnEdge)
        @inbounds begin
        c1,c2 = cellsOnEdge[e]
        inv_dc = inv(dc[e])
        for t in axes(out,3)
            @simd ivdep for k in axes(out,1)
                out[k,e,t] = muladd(inv_dc,(c_field[k,c2,t] - c_field[k,c1,t]),out[k,e,t])
            end
        end
        end #inbounds
    end
    return out
end

function (∇e::GradientAtEdge)(e_field::AbstractArray,c_field::AbstractArray)
    is_proper_size(c_field,∇e.nCells) || throw(DomainError(c_field,"Input array doesn't seem to be a cell field"))
    is_proper_size(e_field,length(∇e.cellsOnEdge)) || throw(DomainError(e_field,"Output array doesn't seem to be an edge field"))

    return gradient_at_edge!(e_field,c_field,∇e.dc,∇e.cellsOnEdge)
end

function (∇e::GradientAtEdge)(e_field::AbstractArray,op::F,c_field::AbstractArray) where F<:Function
    is_proper_size(c_field,∇e.nCells) || throw(DomainError(c_field,"Input array doesn't seem to be a cell field"))
    is_proper_size(e_field,length(∇e.cellsOnEdge)) || throw(DomainError(e_field,"Output array doesn't seem to be an edge field"))

    return gradient_at_edge!(e_field,op,c_field,∇e.dc,∇e.cellsOnEdge)
end

function (∇e::GradientAtEdge)(c_field::AbstractArray)
    is_proper_size(c_field,∇e.nCells) || throw(DomainError(c_field,"Input array doesn't seem to be a cell field"))
    s = construct_new_node_index(size(c_field)...,length(∇e.cellsOnEdge))
    e_field = similar(c_field,s)
    return ∇e(e_field,c_field)
end

struct DivAtCell{N_MAX,TI,TF} <: DifferentialOperator
    n::Int
    indices::Vector{ImmutableVector{N_MAX,TI}}
    weights::Vector{ImmutableVector{N_MAX,TF}}
end

function compute_div_at_cell_weights(areaCell,edgesOnCell::Vector{<:ImmutableVector{N_MAX}},dvEdge::AbstractVector{T},cellsOnEdge) where {N_MAX,T}
    w = Vector{ImmutableVector{N_MAX,T}}(undef,length(areaCell))
    aux = Vector{T}(undef,N_MAX)

    @inbounds for c in eachindex(areaCell)
        inv_a = inv(areaCell[c])

        fill!(aux,zero(T))
        eoc = edgesOnCell[c]
        l = length(eoc)
        for i in Base.OneTo(l)
            e = eoc[i]
            Le = dvEdge[e]
            aux[i] = Le*inv_a*VoronoiMeshDataStruct.sign_edge(cellsOnEdge[e],c)
        end
        w[c] = ImmutableVector{N_MAX}(ntuple(j->getindex(aux,j),Val{N_MAX}()),l)
    end
    return w
end

function DivAtCell(mesh::VoronoiMesh)
    w = compute_div_at_cell_weights(mesh.areaCell,mesh.edgesOnCell,mesh.dvEdge,mesh.cellsOnEdge)
    return DivAtCell(mesh.edges.n,mesh.edgesOnCell,w)
end
