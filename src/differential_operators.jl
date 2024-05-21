struct GradientAtEdge{TI,TF}
    nCells::Int
    dc::Vector{TF}
    cellsOnEdge::Vector{NTuple{2,TI}}
end

GradientAtEdge(mesh::VoronoiMesh) = GradientAtEdge(mesh.cells.n,mesh.edges.dc,mesh.edges.cellsOnEdge)

function gradient_at_edge!(out::AbstractVector,c_field,dc,cellsOnEdge)
    @inbounds for e in eachindex(cellsOnEdge)
        c1,c2 = cellsOnEdge[e]
        out[e] = (c_field[c2] - c_field[c1])/dc[e]
    end
    return out
end

function gradient_at_edge!(out::AbstractVector,op::F,c_field,dc,cellsOnEdge) where F<:Function
    @inbounds for e in eachindex(cellsOnEdge)
        c1,c2 = cellsOnEdge[e]
        out[e] = op(out[e],(c_field[c2] - c_field[c1])/dc[e])
    end
    return out
end

function gradient_at_edge!(out::AbstractMatrix,c_field,dc,cellsOnEdge)
    @inbounds for e in eachindex(cellsOnEdge)
        c1,c2 = cellsOnEdge[e]
        inv_dc = inv(dc[e])
        @simd for k in axes(out,1)
            out[k,e] = inv_dc*(c_field[k,c2] - c_field[k,c1])
        end
    end
    return out
end

function gradient_at_edge!(out::AbstractMatrix,op::F,c_field,dc,cellsOnEdge) where F<:Function
    @inbounds for e in eachindex(cellsOnEdge)
        c1,c2 = cellsOnEdge[e]
        inv_dc = inv(dc[e])
        @simd for k in axes(out,1)
            out[k,e] = op(out[k,e],inv_dc*(c_field[k,c2] - c_field[k,c1]))
        end
    end
    return out
end

function gradient_at_edge!(out::AbstractMatrix,op::typeof(+),c_field,dc,cellsOnEdge)
    @inbounds for e in eachindex(cellsOnEdge)
        c1,c2 = cellsOnEdge[e]
        inv_dc = inv(dc[e])
        @simd for k in axes(out,1)
            out[k,e] = muladd(inv_dc,(c_field[k,c2] - c_field[k,c1]),out[k,e])
        end
    end
    return out
end

function gradient_at_edge!(out::AbstractArray{<:Any,3},c_field,dc,cellsOnEdge)
    @inbounds for e in eachindex(cellsOnEdge)
        c1,c2 = cellsOnEdge[e]
        inv_dc = inv(dc[e])
        for t in axes(out,3)
            @simd for k in axes(out,1)
                out[k,e,t] = inv_dc*(c_field[k,c2,t] - c_field[k,c1,t])
            end
        end
    end
    return out
end

function gradient_at_edge!(out::AbstractArray{<:Any,3},op::F,c_field,dc,cellsOnEdge) where F<:Function
    @inbounds for e in eachindex(cellsOnEdge)
        c1,c2 = cellsOnEdge[e]
        inv_dc = inv(dc[e])
        for t in axes(out,3)
            @simd for k in axes(out,1)
                out[k,e,t] = op(out[k,e,t],inv_dc*(c_field[k,c2,t] - c_field[k,c1,t]))
            end
        end
    end
    return out
end

function gradient_at_edge!(out::AbstractArray{<:Any,3},op::typeof(+),c_field,dc,cellsOnEdge)
    @inbounds for e in eachindex(cellsOnEdge)
        c1,c2 = cellsOnEdge[e]
        inv_dc = inv(dc[e])
        for t in axes(out,3)
            @simd for k in axes(out,1)
                out[k,e,t] = muladd(inv_dc,(c_field[k,c2,t] - c_field[k,c1,t]),out[k,e,t])
            end
        end
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
