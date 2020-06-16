struct Parameters{T<:AbstractFloat}
    W::Array{Array{T,2},1}
    b::Array{Array{T,2},1}
    layer_sizes::Array{Int,1}
    activation_functions::Array{Function,1}
    loss_function::Function

    function Parameters{T}(params_dict::Dict) where {T <: AbstractFloat}
        layer_sizes = params_dict[:layer_sizes]
        input_layer_size = params_dict[:input_layer_size]
        rng = params_dict[:rng]

        L = length(layer_sizes)
        W = Array{Array{T,2},1}(undef, L)
        b = Array{Array{T,2},1}(undef, L)
        for l = 1:L
            if l ≠ 1
                W[l] =
                    randn(rng, layer_sizes[l], layer_sizes[l-1]) .*
                    sqrt(2 / layer_sizes[l-1])
            else
                W[l] =
                    randn(rng, layer_sizes[l], input_layer_size) .*
                    sqrt(2 / input_layer_size)
            end

            b[l] = zeros(layer_sizes[l], 1)
        end

        params_dict[:W] = W
        params_dict[:b] = b
        params_dict[:activation_functions] = get(
            params_dict,
            :activation_functions,
            push!(Function[relu for l = 1:L-1], logistic),
        )
        params_dict[:loss_function] = get(params_dict, :loss_function, cross_entropy_binary)

        return new(
            params_dict[:W],
            params_dict[:b],
            params_dict[:layer_sizes],
            params_dict[:activation_functions],
            params_dict[:loss_function],
        )
    end
end


function initialize_zeros(W_dims, object_types)
    if object_types == :matrices
        return [zeros(Float32, n, m) for (n, m) in W_dims]
    elseif object_types == :vectors
        return [zeros(Float32, n, 1) for (n, m) in W_dims]
    end
end


mutable struct Cache{T<:AbstractFloat}
    A::Array{Array{T,2},1}
    Z::Array{Array{T,2},1}
    dA::Array{Array{T,2},1}
    dZ::Array{Array{T,2},1}
    dW::Array{Array{T,2},1}
    db::Array{Array{T,2},1}
    v_dW::Array{Array{T,2},1}
    v_db::Array{Array{T,2},1}
    s_dW::Array{Array{T,2},1}
    s_db::Array{Array{T,2},1}

    function Cache{T}(params_dict::Dict) where {T <: AbstractFloat}
        # Create array containing the dimensions of the W matrices
        W_dims = pushfirst!(copy(params_dict[:layer_sizes]), params_dict[:input_layer_size])
        W_dims = [(W_dims[l+1], W_dims[l]) for l = 1:length(W_dims)-1]

        # Initialize the cache variables
        A = initialize_zeros(W_dims, :vectors)
        Z = initialize_zeros(W_dims, :vectors)
        dA = initialize_zeros(W_dims, :vectors)
        dZ = initialize_zeros(W_dims, :vectors)
        dW = initialize_zeros(W_dims, :matrices)
        db = initialize_zeros(W_dims, :vectors)

        # v and s are for the Adam and momentum optimization cache
        v_dW = initialize_zeros(W_dims, :matrices)
        v_db = initialize_zeros(W_dims, :vectors)
        s_dW = initialize_zeros(W_dims, :matrices)
        s_db = initialize_zeros(W_dims, :vectors)

        return new(A, Z, dA, dZ, dW, db, v_dW, v_db, s_dW, s_db)
    end
end


mutable struct HyperParameters{T<:AbstractFloat}
    learning_rate::T
    mini_batch_size::Int
    optimization::Symbol
    λ::T
    # Momentum optimization parameters
    β::T
    # Adam optimization parameters
    β₁::T
    β₂::T
    ϵ::T
    t::Int

    function HyperParameters{T}(hparams_dict::Dict) where {T <: AbstractFloat}

        return new(
            convert(T, get(hparams_dict, :learning_rate, 0.01)),
            get(hparams_dict, :mini_batch_size, 64),
            get(hparams_dict, :optimization, :gd),
            convert(T, get(hparams_dict, :λ, 0.1)),
            convert(T, get(hparams_dict, :β, 0.9)),
            convert(T, get(hparams_dict, :β₁, 0.9)),
            convert(T, get(hparams_dict, :β₂, 0.999)),
            convert(T, get(hparams_dict, :ϵ, 1e-8)),
            get(hparams_dict, :t, 0),
        )
    end
end


struct MiniBatch{T<:AbstractFloat}
    X::Array{T,2}
    Y::Array{T,2}
end


mutable struct NeuralNetwork{T<:AbstractFloat}
    X::Array{T,2}
    Y::Array{T,2}
    m::T
    L::Int
    params::Parameters
    hparams::HyperParameters
    cache::Cache
    rng::MersenneTwister
    J::Array{T,1}
    epoch::Int

    function NeuralNetwork{T}(
        X::Array,
        Y::Array;
        params_dict::Dict = Dict(),
        hparams_dict::Dict = Dict(),
    ) where {T <: AbstractFloat}

        # Explicitly typecast to T to ensure all calcs use the same precision
        X = convert(Matrix{T}, X)
        Y = convert(Matrix{T}, Y)

        # Explicitly typecast to Any dicts to avoid type errors
        params_dict = convert(Dict{Any,Any}, params_dict)
        hparams_dict = convert(Dict{Any,Any}, hparams_dict)

        # Set values for useful variables
        params_dict[:input_layer_size] = size(X, 1)
        params_dict[:layer_sizes] = get(params_dict, :layer_sizes, [3, 1])

        # Create the rng and seed it with the value from params_dict
        rng = MersenneTwister(get(params_dict, :seed, 8))
        params_dict[:rng] = rng

        # Initialize instances of custom types
        params = Parameters{T}(params_dict)
        hparams = HyperParameters{T}(hparams_dict)
        cache = Cache{T}(params_dict)

        # Set starting values of variables
        m = convert(T, size(X, 2))
        L = length(params_dict[:layer_sizes])
        J = []
        epoch = 0

        NN = new{T}(X, Y, m, L, params, hparams, cache, rng, J, epoch)
    end
end
