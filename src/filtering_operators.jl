abstract type FilteringOperator <: VoronoiOperator end

using OrderedCollections

struct CellBoxFilter{N_MAX,TI,TF} <: FilteringOperator
    length::TF
    weights::Vector{ImmutableVector{N_MAX,TF}}
    indices::Vector{ImmutableVector{N_MAX,TI}}
end

function in_circle(c,r2,p)
    cp = p-c
    return (cp⋅cp) <= r2
end

function weight_indices_matrix_to_immutable(::Val{N_MAX},nElements::AbstractVector,inds::Matrix{TI},wm::Matrix{TF}) where {N_MAX,TI,TF}
    nCells = length(nElements)
    w = Vector{ImmutableVector{N_MAX,TF}}(undef,nCells)
    indices = Vector{ImmutableVector{N_MAX,TI}}(undef,nCells)

    @inbounds for i in Base.OneTo(nCells)
        r = Base.OneTo(nElements[i])
        w[i] = ImmutableVector{N_MAX}(@view wm[r,i])
        indices[i] = ImmutableVector{N_MAX}(@view inds[r,i])
    end

    return indices, w
end

function compute_cell_box_filter_weights_and_indices_periodic(Δ,c_position,areaCell,cellsOnCell::AbstractVector{<:ImmutableVector{N_MAX}},verticesOnCell,v_position,xp::Number,yp::Number) where N_MAX
    nCells = length(areaCell)
    w = zeros(eltype(areaCell),255,nCells) # I'm assuming the filtering stencil won't be larger than 255, which is the maximum number of elements supported by ImmutableVectors
    inds = zeros(Int,255,nCells)
    nElementsOnCell = zeros(Int,nCells)
    r = Δ/2
    r2 = r*r

    checked_cells = OrderedSet{Int}() # To store cells that were already checked
    cells_to_check = OrderedSet{Int}() # Set of cells we want to check
    neighbour_cells = OrderedSet{Int}() # Set of cells surrounding a given cell

    for c in Base.OneTo(nCells)
        empty!(checked_cells)
        center = c_position[c]
        current_vertices = verticesOnCell[c]
        n_vertices_on_cell = length(current_vertices)
        f1 = (v) -> closest(center,v_position[v],xp,yp)
        f2 = (vp) -> in_circle(center,r2,vp)


        current_vertices_position = map(f1,current_vertices) # Vector with vertices positions adjusted to periodicity
        vertices_in_circle = map(f2,current_vertices_position) # Vector of Bools telling whether vertex at current_vertices_position[i] is inside the disk or not
        n_vertices_in_circle = sum(vertices_in_circle) # Since true == 1 and false == 0

        if n_vertices_in_circle == 0
            w[1,c] = 1.0
            inds[1,c] = c
            nElementsOnCell[c] = 1
            continue
        end

        if (n_vertices_in_circle != n_vertices_on_cell)
            error("Haven't coded it yet")
        end
        
        i=1
        w[i,c] = areaCell[c]
        inds[i,c] = c
        push!(checked_cells,c)

        union!(cells_to_check,cellsOnCell[c])
        #Loop over surrounding cells until no cell has any vertex inside de disk 
        while !isempty(cells_to_check)
            current_cell = popfirst!(cells_to_check)
            current_vertices = verticesOnCell[current_cell]
            n_vertices_on_cell = length(current_vertices)

            current_vertices_position = map(f1,current_vertices) # Vector with vertices positions adjusted to periodicity
            vertices_in_circle = map(f2,current_vertices_position) # Vector of Bools telling whether vertex at current_vertices_position[i] is inside the disk or not
            n_vertices_in_circle = sum(vertices_in_circle) # Since true == 1 and false == 0

            if n_vertices_in_circle != 0
                i+=1

                inds[i,c] = current_cell
                if n_vertices_in_circle == n_vertices_on_cell
                    w[i,c] = areaCell[current_cell]
                else
                    w[i,c] = polygon_circle_intersection_area(center,r2,current_vertices_position,vertices_in_circle)
                end
                empty!(neighbour_cells)
                union!(neighbour_cells,cellsOnCell[current_cell])
                union!(cells_to_check,setdiff!(neighbour_cells,checked_cells))
            end
            push!(checked_cells,current_cell)
        end

        nElementsOnCell[c] = i
        w[:,c] ./= sum(@view w[:,c])
    end

    n_max = maximum(nElementsOnCell)
    #Avoid dynamic dispatch for most common cases
    if n_max == (N_MAX + 1)
        return weight_indices_matrix_to_immutable(Val{N_MAX+1}(),nElementsOnCell,inds,w)
    elseif n_max == (2*N_MAX+1)
        return weight_indices_matrix_to_immutable(Val{2*N_MAX+1}(),nElementsOnCell,inds,w)
    elseif n_max == (3*N_MAX+1)
        return weight_indices_matrix_to_immutable(Val{3*N_MAX+1}(),nElementsOnCell,inds,w)
    elseif n_max == (4*N_MAX+1)
        return weight_indices_matrix_to_immutable(Val{4*N_MAX+1}(),nElementsOnCell,inds,w)
    else
        return weight_indices_matrix_to_immutable(Val{n_max}(),nElementsOnCell,inds,w)
    end
end

function CellBoxFilter(mesh::VoronoiMesh{false},Δ::Number)
    indices,w = compute_cell_box_filter_weights_and_indices_periodic(Δ,mesh.cells.position,mesh.cells.area,mesh.cellsOnCell,mesh.verticesOnCell,mesh.vertices.position,mesh.attributes[:x_period]::Float64,mesh.attributes[:y_period]::Float64)
    return CellBoxFilter(Δ,w,indices)
end

function CellBoxFilter(mesh::VoronoiMesh{false})
    Δ = 2*mesh.attributes[:dc]::Float64
    return CellBoxFilter(mesh,Δ)
end

function (cellFilter::CellBoxFilter)(c_field::AbstractArray,e_field::AbstractArray)
    is_proper_size(e_field,length(cellFilter.weights)) || throw(DomainError(c_field,"Input array doesn't seem to be a cell field"))
    is_proper_size(c_field,length(cellFilter.weights)) || throw(DomainError(c_field,"Output array doesn't seem to be a cell field"))

    weighted_sum_transformation!(c_field,e_field,cellFilter.weights, cellFilter.indices)
    
    return c_field
end

function (cellFilter::CellBoxFilter)(e_field::AbstractArray)
    is_proper_size(e_field,length(cellFilter.weights)) || throw(DomainError(c_field,"Input array doesn't seem to be a cell field"))
    c_field = similar(e_field,Base.promote_op(*,eltype(eltype(cellFilter.weights)),eltype(e_field)))
    return cellFilter(c_field,e_field)
end

function (cellFilter::CellBoxFilter)(c_field::AbstractArray,op::F,e_field::AbstractArray) where {F<:Function}
    is_proper_size(e_field,length(cellFilter.weights)) || throw(DomainError(c_field,"Input array doesn't seem to be a cell field"))
    is_proper_size(c_field,length(cellFilter.weights)) || throw(DomainError(c_field,"Output array doesn't seem to be a cell field"))

    weighted_sum_transformation!(c_field, op, e_field, cellFilter.weights, cellFilter.indices)

    return c_field
end
