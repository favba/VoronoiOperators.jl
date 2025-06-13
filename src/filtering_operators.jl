abstract type FilteringOperator <: LinearVoronoiOperator end

using OrderedCollections

struct CellBoxFilter{N_MAX, TI, TF, TW <: Union{TF, Vector{TF}}} <: FilteringOperator
    weights::Vector{ImmutableVector{N_MAX, TF}}
    indices::ImVecArray{N_MAX, TI, 1}
    width::TW
end
name_input(::CellBoxFilter) = "cell"
name_output(::CellBoxFilter) = "cell"
n_input(a::CellBoxFilter) = length(a.indices)

function in_circle(c, r2, p)
    cp = p - c
    return (cp ⋅ cp) <= r2
end

function weight_indices_matrix_to_immutable(::Val{N_MAX}, nElements::AbstractVector, inds::Matrix{TI}, wm::Matrix{TF}) where {N_MAX, TI, TF}
    nCells = length(nElements)
    w = Vector{ImmutableVector{N_MAX, TF}}(undef, nCells)
    indices = ImmutableVectorArray(Vector{NTuple{N_MAX, TI}}(undef,nCells), Vector{UInt8}(undef, nCells))

    @batch for i in Base.OneTo(nCells)
        @inbounds begin
            r = Base.OneTo(nElements[i])
            w[i] = ImmutableVector{N_MAX}(@view wm[r, i])
            indices[i] = ImmutableVector{N_MAX}(@view inds[r, i])
        end
    end

    return indices, w
end

function compute_cell_box_filter_weights_and_indices_periodic(Δ::Number, c_position, areaCell, cellsOnCell::AbstractVector{<:ImmutableVector{N_MAX}}, verticesOnCell, v_position, xp::Number, yp::Number) where {N_MAX}
    nCells = length(areaCell)
    w = zeros(eltype(areaCell), 255, nCells) # I'm assuming the filtering stencil won't be larger than 255, which is the maximum number of elements supported by ImmutableVectors
    IT = eltype(eltype(cellsOnCell))
    inds = zeros(IT, 255, nCells)
    nElementsOnCell = zeros(IT, nCells)
    r = Δ / 2
    r2 = r * r

    checked_cells_task = TaskLocalValue{OrderedSet{IT}}(() -> OrderedSet{IT}()) # To store cells that were already checked
    cells_to_check_task = TaskLocalValue{OrderedSet{IT}}(() -> OrderedSet{IT}()) # Set of cells we want to check
    neighbour_cells_task = TaskLocalValue{OrderedSet{IT}}(() -> OrderedSet{IT}()) # Set of cells surrounding a given cell

    @parallel for c in Base.OneTo(nCells)
        checked_cells = checked_cells_task[]
        cells_to_check = cells_to_check_task[]
        neighbour_cells = neighbour_cells_task[]

        empty!(checked_cells)
        center = c_position[c]
        current_vertices = verticesOnCell[c]
        n_vertices_on_cell = length(current_vertices)
        f1 = (v) -> closest(center, v_position[v], xp, yp)
        f2 = (vp) -> in_circle(center, r2, vp)


        current_vertices_position = map(f1, current_vertices) # Vector with vertices positions adjusted to periodicity
        vertices_in_circle = map(f2, current_vertices_position) # Vector of Bools telling whether vertex at current_vertices_position[i] is inside the disk or not
        n_vertices_in_circle = sum(vertices_in_circle) # Since true == 1 and false == 0

        if n_vertices_in_circle == 0
            w[1, c] = 1.0
            inds[1, c] = c
            nElementsOnCell[c] = 1
            continue
        end

        if (n_vertices_in_circle != n_vertices_on_cell)
            error("Haven't coded it yet")
        end

        i = 1
        w[i, c] = areaCell[c]
        inds[i, c] = c
        push!(checked_cells, c)

        union!(cells_to_check, cellsOnCell[c])
        #Loop over surrounding cells until no cell has any vertex inside de disk
        while !isempty(cells_to_check)
            current_cell = popfirst!(cells_to_check)
            current_vertices = verticesOnCell[current_cell]
            n_vertices_on_cell = length(current_vertices)

            current_vertices_position = map(f1, current_vertices) # Vector with vertices positions adjusted to periodicity
            vertices_in_circle = map(f2, current_vertices_position) # Vector of Bools telling whether vertex at current_vertices_position[i] is inside the disk or not
            n_vertices_in_circle = sum(vertices_in_circle) # Since true == 1 and false == 0

            if n_vertices_in_circle != 0
                i += 1

                inds[i, c] = current_cell
                if n_vertices_in_circle == n_vertices_on_cell
                    w[i, c] = areaCell[current_cell]
                else
                    w[i, c] = polygon_circle_intersection_area(center, r2, current_vertices_position, vertices_in_circle)
                end
                empty!(neighbour_cells)
                union!(neighbour_cells, cellsOnCell[current_cell])
                union!(cells_to_check, setdiff!(neighbour_cells, checked_cells))
            end
            push!(checked_cells, current_cell)
        end

        nElementsOnCell[c] = i
        w[:, c] ./= sum(@view w[:, c])
    end

    n_max = maximum(nElementsOnCell)
    #Avoid dynamic dispatch for most common cases
    if n_max == (N_MAX + 1)
        return weight_indices_matrix_to_immutable(Val{N_MAX + 1}(), nElementsOnCell, inds, w)
    elseif n_max == (2 * N_MAX + 1)
        return weight_indices_matrix_to_immutable(Val{2 * N_MAX + 1}(), nElementsOnCell, inds, w)
    elseif n_max == (3 * N_MAX + 1)
        return weight_indices_matrix_to_immutable(Val{3 * N_MAX + 1}(), nElementsOnCell, inds, w)
    elseif n_max == (4 * N_MAX + 1)
        return weight_indices_matrix_to_immutable(Val{4 * N_MAX + 1}(), nElementsOnCell, inds, w)
    else
        return weight_indices_matrix_to_immutable(Val{Int(n_max)}(), nElementsOnCell, inds, w)
    end
end

function CellBoxFilter(mesh::AbstractVoronoiMesh{false}, Δ::Number)
    indices, w = compute_cell_box_filter_weights_and_indices_periodic(Δ, mesh.cells.position, mesh.cells.area, mesh.cells.cells, mesh.cells.vertices, mesh.vertices.position, mesh.x_period, mesh.y_period)
    return CellBoxFilter(w, indices, Δ)
end

function compute_cell_box_filter_weights_and_indices_periodic_variable_resolution(width_func::F, c_position, areaCell, cellsOnCell::AbstractVector{<:ImmutableVector{N_MAX}}, verticesOnCell, v_position, xp::Number, yp::Number) where {F <: Function, N_MAX}
    nCells = length(areaCell)
    w = zeros(eltype(areaCell), 255, nCells) # I'm assuming the filtering stencil won't be larger than 255, which is the maximum number of elements supported by ImmutableVectors
    width = zeros(eltype(areaCell), nCells)
    IT = eltype(eltype(cellsOnCell))
    inds = zeros(IT, 255, nCells)
    nElementsOnCell = zeros(IT, nCells)

    checked_cells = OrderedSet{IT}() # To store cells that were already checked
    cells_to_check = OrderedSet{IT}() # Set of cells we want to check
    neighbour_cells = OrderedSet{IT}() # Set of cells surrounding a given cell

    for c in Base.OneTo(nCells)
        Δ = width_func(c)
        width[c] = Δ
        r = Δ / 2
        r2 = r * r
        empty!(checked_cells)
        center = c_position[c]
        current_vertices = verticesOnCell[c]
        n_vertices_on_cell = length(current_vertices)
        f1 = (v) -> closest(center, v_position[v], xp, yp)
        f2 = (vp) -> in_circle(center, r2, vp)


        current_vertices_position = map(f1, current_vertices) # Vector with vertices positions adjusted to periodicity
        vertices_in_circle = map(f2, current_vertices_position) # Vector of Bools telling whether vertex at current_vertices_position[i] is inside the disk or not
        n_vertices_in_circle = sum(vertices_in_circle) # Since true == 1 and false == 0

        if n_vertices_in_circle == 0
            w[1, c] = 1.0
            inds[1, c] = c
            nElementsOnCell[c] = 1
            continue
        end

        if (n_vertices_in_circle != n_vertices_on_cell)
            error("Haven't coded it yet")
        end

        i = 1
        w[i, c] = areaCell[c]
        inds[i, c] = c
        push!(checked_cells, c)

        union!(cells_to_check, cellsOnCell[c])
        #Loop over surrounding cells until no cell has any vertex inside de disk
        while !isempty(cells_to_check)
            current_cell = popfirst!(cells_to_check)
            current_vertices = verticesOnCell[current_cell]
            n_vertices_on_cell = length(current_vertices)

            current_vertices_position = map(f1, current_vertices) # Vector with vertices positions adjusted to periodicity
            vertices_in_circle = map(f2, current_vertices_position) # Vector of Bools telling whether vertex at current_vertices_position[i] is inside the disk or not
            n_vertices_in_circle = sum(vertices_in_circle) # Since true == 1 and false == 0

            if n_vertices_in_circle != 0
                i += 1

                inds[i, c] = current_cell
                if n_vertices_in_circle == n_vertices_on_cell
                    w[i, c] = areaCell[current_cell]
                else
                    w[i, c] = polygon_circle_intersection_area(center, r2, current_vertices_position, vertices_in_circle)
                end
                empty!(neighbour_cells)
                union!(neighbour_cells, cellsOnCell[current_cell])
                union!(cells_to_check, setdiff!(neighbour_cells, checked_cells))
            end
            push!(checked_cells, current_cell)
        end

        nElementsOnCell[c] = i
        w[:, c] ./= sum(@view w[:, c])
    end

    n_max = maximum(nElementsOnCell)
    #Avoid dynamic dispatch for most common cases
    if n_max == (N_MAX + 1)
        return (weight_indices_matrix_to_immutable(Val{N_MAX + 1}(), nElementsOnCell, inds, w)..., width)
    elseif n_max == (2 * N_MAX + 1)
        return (weight_indices_matrix_to_immutable(Val{2 * N_MAX + 1}(), nElementsOnCell, inds, w)..., width)
    elseif n_max == (3 * N_MAX + 1)
        return (weight_indices_matrix_to_immutable(Val{3 * N_MAX + 1}(), nElementsOnCell, inds, w)..., width)
    elseif n_max == (4 * N_MAX + 1)
        return (weight_indices_matrix_to_immutable(Val{4 * N_MAX + 1}(), nElementsOnCell, inds, w)..., width)
    else
        return (weight_indices_matrix_to_immutable(Val{Int(n_max)}(), nElementsOnCell, inds, w)..., width)
    end
end

function CellBoxFilter(mesh::AbstractVoronoiMesh{false}, f::Function)
    indices, w, Δ = compute_cell_box_filter_weights_and_indices_periodic_variable_resolution(f, mesh.cells.position, mesh.cells.area, mesh.cells.cells, mesh.cells.vertices, mesh.vertices.position, mesh.x_period, mesh.y_period)
    return CellBoxFilter(w, indices, Δ)
end

function CellBoxFilter(mesh::AbstractVoronoiMesh{false}, variable_resolution::Bool = false, ratio = 2.0)
    if !variable_resolution
        #Δ = 2 * mesh.attributes[:dc]::Float64
        Δ = 2 * (sum(mesh.edges.lengthDual) / mesh.edges.n)
        return CellBoxFilter(mesh, Δ)
    else
        f = let dcEdge = mesh.edges.lengthDual, edgesOnCell = mesh.cells.edges
            @inline function (c)
                edges = edgesOnCell[c]
                dcs = map(e -> dcEdge[e], edges)
                Δ = ratio * (sum(dcs) / length(dcs)) # filter width is is `ratio*mean_dc`
                return Δ
            end
        end
        return CellBoxFilter(mesh, f)
    end
end
