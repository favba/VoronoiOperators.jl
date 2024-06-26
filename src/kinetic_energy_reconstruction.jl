abstract type CellKineticEnergyReconstruction{N_MAX,TI,TF} <: VoronoiOperator end

struct CellKineticEnergyRingler{N_MAX,TI,TF} <: CellKineticEnergyReconstruction{N_MAX,TI,TF}
    nEdges::Int
    edgesOnCell::Vector{ImmutableVector{N_MAX,TI}}
    weights::Vector{ImmutableVector{N_MAX,TF}}
end

function compute_weights_ringler_kinetic_energy!(w::Vector{ImmutableVector{N_MAX,T}},aC,Le,dc,edgesOnCell::Vector{<:ImmutableVector{N_MAX}}) where {T,N_MAX}
    aux = Vector{T}(undef,N_MAX)

    @inbounds for c in eachindex(edgesOnCell)
        term = inv(aC[c]*4)
        fill!(aux,zero(T))

        eoc = edgesOnCell[c]
        l = length(eoc)
        for i in Base.OneTo(l)
            e = eoc[i]
            aux[i] = term*Le[e]*dc[e]
        end
        w[c] = ImmutableVector{N_MAX}(ntuple(j->getindex(aux,j),Val{N_MAX}()),l)
    end
    return w
end

function compute_weights_ringler_kinetic_energy(aC::Vector{T},Le,dc,edgesOnCell::Vector{<:ImmutableVector{N_MAX}}) where {T,N_MAX}
    w = Vector{ImmutableVector{N_MAX,T}}(undef,length(edgesOnCell))
    return compute_weights_ringler_kinetic_energy!(w,aC,Le,dc,edgesOnCell)
end

function CellKineticEnergyRingler(cells,edges)
    w = compute_weights_ringler_kinetic_energy(cells.area,edges.dv,edges.dc,cells.indices.edges)
    return CellKineticEnergyRingler(edges.n,cells.indices.edges,w)
end

CellKineticEnergyRingler(m::VoronoiMesh) = CellKineticEnergyRingler(m.cells,m.edges)

function (kc::CellKineticEnergyRingler)(c_field::AbstractArray,e_field::AbstractArray)
    is_proper_size(e_field,kc.nEdges) || throw(DomainError(e_field,"Input array doesn't seem to be an edge field"))
    is_proper_size(c_field,length(kc.edgesOnCell)) || throw(DomainError(c_field,"Output array doesn't seem to be a cell field"))

    square = @inline function (x); x*x;end
    weighted_sum_transformation!(c_field,e_field,square,kc.weights, kc.edgesOnCell)
    
    return c_field
end

function (kc::CellKineticEnergyRingler)(e_field::AbstractArray)
    is_proper_size(e_field,kc.nEdges) || throw(DomainError(e_field,"Input array doesn't seem to be an edge field"))
    s = construct_new_node_index(size(e_field)...,length(kc.edgesOnCell))
    c_field = similar(e_field,Base.promote_op(*,eltype(eltype(kc.weights)),eltype(e_field)),s)
    return kc(c_field,e_field)
end

function (kc::CellKineticEnergyRingler)(c_field::AbstractArray,op::F,e_field::AbstractArray) where {F<:Function}
    is_proper_size(e_field,kc.nEdges) || throw(DomainError(e_field,"Input array doesn't seem to be an edge field"))
    is_proper_size(c_field,length(kc.edgesOnCell)) || throw(DomainError(c_field,"Output array doesn't seem to be a cell field"))

    square = @inline function (x); x*x;end
    weighted_sum_transformation!(c_field,op,e_field,square,kc.weights, kc.edgesOnCell)
 
    return c_field
end

abstract type VertexKineticEnergyReconstruction{TI,TF} end

struct VertexKineticEnergyGassmann{TI,TF} <: VertexKineticEnergyReconstruction{TI,TF}
    nEdges::Int
    edgesOnVertex::Vector{NTuple{3,TI}}
    weights::Vector{NTuple{3,TF}}
end

function compute_weights_vertex_kinetic_energy!(w,aV,Le,dc,edgesOnVertex)

    @inbounds for v in eachindex(edgesOnVertex)
        e1,e2,e3 = edgesOnVertex[v]
        term = inv(aV[v]*4)

        w1 = term*Le[e1]*dc[e1]
        w2 = term*Le[e2]*dc[e2]
        w3 = term*Le[e3]*dc[e3]

        w[v] = (w1,w2,w3)
    end
    return w
end

function compute_weights_vertex_kinetic_energy(aV,Le,dc,edgesOnVertex)
    w = Vector{NTuple{3,eltype(aV)}}(undef,length(edgesOnVertex))
    return compute_weights_vertex_kinetic_energy!(w,aV,Le,dc,edgesOnVertex)
end

function VertexKineticEnergyGassmann(vertices,edges)
    w = compute_weights_vertex_kinetic_energy(vertices.area,edges.dv,edges.dc,vertices.indices.edges)
    return VertexKineticEnergyGassmann(edges.n,vertices.indices.edges,w)
end

VertexKineticEnergyGassmann(m::VoronoiMesh) = VertexKineticEnergyGassmann(m.vertices,m.edges)

function (kv::VertexKineticEnergyGassmann)(v_field::AbstractArray,e_field::AbstractArray)
    is_proper_size(e_field,kv.nEdges) || throw(DomainError(e_field,"Input array doesn't seem to be an edge field"))
    is_proper_size(v_field,length(kv.edgesOnVertex)) || throw(DomainError(v_field,"Output array doesn't seem to be a vertex field"))

    square = @inline function (x); x*x;end
    weighted_sum_transformation!(v_field,e_field,square,kv.weights, kv.edgesOnVertex)
    
    return v_field
end

function (kv::VertexKineticEnergyGassmann)(e_field::AbstractArray)
    is_proper_size(e_field,kv.nEdges) || throw(DomainError(e_field,"Input array doesn't seem to be an edge field"))
    s = construct_new_node_index(size(e_field)...,length(kv.edgesOnVertex))
    v_field = similar(e_field,Base.promote_op(*,eltype(eltype(kv.weights)),eltype(e_field)),s)
    return kv(v_field,e_field)
end

function (kv::VertexKineticEnergyGassmann)(v_field::AbstractArray,op::F,e_field::AbstractArray) where {F<:Function}
    is_proper_size(e_field,kv.nEdges) || throw(DomainError(e_field,"Input array doesn't seem to be an edge field"))
    is_proper_size(v_field,length(kv.edgesOnVertex)) || throw(DomainError(v_field,"Output array doesn't seem to be a vertex field"))

    square = @inline function (x); x*x;end
    weighted_sum_transformation!(v_field,op,e_field,square,kv.weights, kv.edgesOnVertex)
 
    return v_field
end

function compute_weights_vertex_kinetic_energy_modified!(w,aV,Lev,dc,edgesOnVertex)

    @inbounds for v in eachindex(edgesOnVertex)
        e1,e2,e3 = edgesOnVertex[v]
        term = inv(aV[v]*2)
        Lev1,Lev2,Lev3 = Lev[v]

        w1 = term*Lev1*dc[e1]
        w2 = term*Lev2*dc[e2]
        w3 = term*Lev3*dc[e3]

        w[v] = (w1,w2,w3)
    end
    return w
end

function compute_vertex_edge_distance_periodic!(Lev,vpos,epos,edgesOnVertex,xp::Number,yp::Number)
    @inbounds for v in eachindex(edgesOnVertex)
        e1,e2,e3 = edgesOnVertex[v]
        vp = vpos[v]
        e1pos = closest(vp,epos[e1],xp,yp)
        Lev1 = norm(e1pos-vp)
        e2pos = closest(vp,epos[e2],xp,yp)
        Lev2 = norm(e2pos-vp)
        e3pos = closest(vp,epos[e3],xp,yp)
        Lev3 = norm(e3pos-vp)
        Lev[v] = (Lev1,Lev2,Lev3)
    end
    return Lev
end

struct CellKineticEnergyMPAS{N_MAX,TI,TF} <: CellKineticEnergyReconstruction{N_MAX,TI,TF}
    vertexReconstruction::VertexKineticEnergyGassmann{TI,TF}
    RinglerReconstruction::CellKineticEnergyRingler{N_MAX,TI,TF}
    weightsVertexToCell::Vector{ImmutableVector{N_MAX,TF}}
    verticesOnCell::Vector{ImmutableVector{N_MAX,TI}}
    alpha::TF
    kv1d::Base.RefValue{Vector{TF}}
    kv2d::Base.RefValue{Matrix{TF}}
    kv3d::Base.RefValue{Array{TF,3}}
end

function compute_vertex_to_cell_weight!(w::Vector{ImmutableVector{N_MAX,TF}},verticesOnCell,areaCell,kiteAreaOnVertex,cellsOnVertex) where {N_MAX,TF}
    aux = Vector{TF}(undef,N_MAX)
    @inbounds for c in eachindex(areaCell)
        Ac = areaCell[c]
        fill!(aux,zero(TF))
        voc = verticesOnCell[c]
        l = length(voc)
        for i in Base.OneTo(l)
            v = voc[i]
            aux[i] = select_kite_area(kiteAreaOnVertex,cellsOnVertex,v,c)/Ac
        end
        w[c] = ImmutableVector{N_MAX}(ntuple(j->getindex(aux,j),Val{N_MAX}()),l)
    end
    return w
end

function compute_vertex_to_cell_weight(mesh::VoronoiMesh)
    w = Vector{ImmutableVector{max_n_edges(typeof(mesh.cells)),eltype(mesh.cells.area)}}(undef,mesh.cells.n)
    return compute_vertex_to_cell_weight!(w,mesh.verticesOnCell,mesh.areaCell,mesh.vertices.kiteAreas,mesh.cellsOnVertex)
end

function CellKineticEnergyMPAS(mesh::VoronoiMesh,alpha = 1 - 0.375) 
    T = float_precision(typeof(mesh.cells))
    return CellKineticEnergyMPAS(VertexKineticEnergyGassmann(mesh),CellKineticEnergyRingler(mesh),compute_vertex_to_cell_weight(mesh),mesh.cells.indices.vertices,alpha,Ref{Vector{T}}(),Ref{Matrix{T}}(),Ref{Array{T,3}}())
end

function get_proper_kv(ckm::CellKineticEnergyMPAS,u::Vector{TF}) where TF
    if !isassigned(ckm.kv1d)
        ckm.kv1d[] = Vector{TF}(undef,length(ckm.vertexReconstruction.edgesOnVertex))
    end
    return ckm.kv1d[]
end

function get_proper_kv(ckm::CellKineticEnergyMPAS,u::Matrix{TF}) where TF
    if !isassigned(ckm.kv2d)
        ckm.kv2d[] = Matrix{TF}(undef,size(u,1),length(ckm.vertexReconstruction.edgesOnVertex))
    end
    return ckm.kv2d[]
end

function get_proper_kv(ckm::CellKineticEnergyMPAS,u::Array{TF,3}) where TF
    if !isassigned(ckm.kv3d)
        ckm.kv3d[] = Array{TF,3}(undef,size(u,1),length(ckm.vertexReconstruction.edgesOnVertex),size(u,3))
    end
    return ckm.kv3d[]
end

function (ckm::CellKineticEnergyMPAS)(c_field::AbstractArray,u::AbstractArray)
    is_proper_size(c_field,length(ckm.weightsVertexToCell)) || throw(DomainError(c_field,"Output array doesn't seem to be a cell field"))
    is_proper_size(u,ckm.RinglerReconstruction.nEdges) || throw(DomainError(u,"Input array doesn't seem to be an edge field"))

    kv = get_proper_kv(ckm,u)
    ckm.vertexReconstruction(kv,u)
    ckm.RinglerReconstruction(c_field,u)

    f = let α=ckm.alpha, β = 1.0 - α
        @inline function (x,y); α*x + β*y;end
    end

    weighted_sum_transformation!(c_field,f,kv,ckm.weightsVertexToCell,ckm.verticesOnCell)

    return c_field
end

function (kc::CellKineticEnergyMPAS)(e_field::AbstractArray)
    is_proper_size(e_field,kc.RinglerReconstruction.nEdges) || throw(DomainError(e_field,"Input array doesn't seem to be an edge field"))
    s = construct_new_node_index(size(e_field)...,length(kc.RinglerReconstruction.edgesOnCell))
    c_field = similar(e_field,Base.promote_op(*,eltype(eltype(kc.RinglerReconstruction.weights)),eltype(e_field)),s)
    return kc(c_field,e_field)
end

function (ckm::CellKineticEnergyMPAS)(c_field::AbstractArray,op::F,u::AbstractArray) where F<:Union{typeof(Base.:+),typeof(Base.:-)}
    is_proper_size(c_field,length(ckm.weightsVertexToCell)) || throw(DomainError(c_field,"Output array doesn't seem to be a cell field"))
    is_proper_size(u,ckm.RinglerReconstruction.nEdges) || throw(DomainError(u,"Input array doesn't seem to be an edge field"))

    kv = get_proper_kv(ckm,u)
    ckm.vertexReconstruction(kv,u)

    f1 = let op=op, α = ckm.alpha
        @inline function (x,y); op(x,α*y);end
    end

    ckm.RinglerReconstruction(c_field, f1, u)

    f2 = let op=op, β = 1.0 - ckm.alpha
        @inline function (x,y);  op(x,β*y);end
    end

    weighted_sum_transformation!(c_field,f2,kv,ckm.weightsVertexToCell,ckm.verticesOnCell)

    return c_field
end

struct CellKineticEnergyVelRecon{N_MAX,TI,TF,TR<:CellVelocityReconstruction{N_MAX,TI,TF}} <: CellKineticEnergyReconstruction{N_MAX,TI,TF}
    uR::TR
end

CellKineticEnergyPerot(mesh::VoronoiMesh) = CellKineticEnergyVelRecon(CellVelocityReconstructionPerot(mesh))

function (kc::CellKineticEnergyVelRecon)(c_field::AbstractArray,e_field::AbstractArray)
    is_proper_size(e_field,kc.uR.nEdges) || throw(DomainError(e_field,"Input array doesn't seem to be an edge field"))
    is_proper_size(c_field,length(kc.uR.edgesOnCell)) || throw(DomainError(c_field,"Output array doesn't seem to be a cell field"))

    energy = @inline function (y,x); 0.5*(x⋅x);end
    kc.uR(c_field,energy,e_field)
    
    return c_field
end

function (kc::CellKineticEnergyVelRecon)(e_field::AbstractArray)
    is_proper_size(e_field,kc.uR.nEdges) || throw(DomainError(e_field,"Input array doesn't seem to be an edge field"))
    s = construct_new_node_index(size(e_field)...,length(kc.uR.edgesOnCell))
    c_field = similar(e_field,Base.promote_op(dot,eltype(eltype(kc.uR.weights)),eltype(eltype(kc.uR.weights))),s)
    return kc(c_field,e_field)
end

function (kc::CellKineticEnergyVelRecon)(c_field::AbstractArray,op::F,e_field::AbstractArray) where {F<:Function}
    is_proper_size(e_field,kc.uR.nEdges) || throw(DomainError(e_field,"Input array doesn't seem to be an edge field"))
    is_proper_size(c_field,length(kc.uR.edgesOnCell)) || throw(DomainError(c_field,"Output array doesn't seem to be a cell field"))

    energy = @inline function (y,x); op(y,0.5*(x⋅x));end
    kc.uR(c_field,energy,e_field)
 
    return c_field
end
