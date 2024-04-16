@inline get_node_index(i::Integer) = i
@inline get_node_index(k::Integer,i::Integer) = i
@inline get_node_index(k::Integer,i::Integer,t::Integer) = i

@inline construct_new_node_index(i::Integer,n::Integer) = (oftype(i,n),)
@inline construct_new_node_index(k::Integer,i::Integer,n::Integer) = (k,oftype(i,n))
@inline construct_new_node_index(k::Integer,i::Integer,t::Integer,n::Integer) = (k,oftype(i,n),t)

@inline is_proper_size(field::AbstractVector,n::Integer) = length(field) == n
@inline is_proper_size(field::AbstractMatrix,n::Integer) = size(field,2) == n
@inline is_proper_size(field::AbstractArray{<:Any,3},n::Integer) = size(field,2) == n
