mutable struct Parameters
	W                    ::Array{Array{Float32, 2}, 1}
	b                    ::Array{Array{Float32, 2}, 1}
	layer_sizes          ::Array{Int, 1}
	activation_functions ::Array{String, 1}

	function Parameters(params_dict)
 		initialize_params!(params_dict)

		return new(
			params_dict["W"],
			params_dict["b"],
			params_dict["layer_sizes"],
			params_dict["activation_functions"])


function initialize_zeros(W_dims, object_types)
	if object_types == "matrices"
		return [zeros(Float32, n, m) for (n, m) in W_dims]
	elseif object_types == "vectors"
		return [zeros(Float32, n, 1) for (n, m) in W_dims]
	end
end


mutable struct Cache
	A    ::Array{Array{Float32, 2}, 1}
	Z    ::Array{Array{Float32, 2}, 1}
	dA   ::Array{Array{Float32, 2}, 1}
	dZ   ::Array{Array{Float32, 2}, 1}
	dW   ::Array{Array{Float32, 2}, 1}
	db   ::Array{Array{Float32, 2}, 1}
	v_dW ::Array{Array{Float32, 2}, 1}
	v_db ::Array{Array{Float32, 2}, 1}
	s_dW ::Array{Array{Float32, 2}, 1}
	s_db ::Array{Array{Float32, 2}, 1}

	function Cache(params_dict)
		# Create array containing the dimensions of the W matrices
		W_dims = pushfirst!(copy(params_dict["layer_sizes"]), params_dict["input_layer_size"])
		W_dims = [(W_dims[l+1], W_dims[l]) for l in 1:length(W_dims)-1]

		# Initialize the cache variables
		A    = initialize_zeros(W_dims, "vectors")
		Z    = initialize_zeros(W_dims, "vectors")
		dA   = initialize_zeros(W_dims, "vectors")
		dZ   = initialize_zeros(W_dims, "vectors")
		dW   = initialize_zeros(W_dims, "matrices")
		db   = initialize_zeros(W_dims, "vectors")

		# v and s are for the Adam and momentum optimization cache
		v_dW = initialize_zeros(W_dims, "matrices")
		v_db = initialize_zeros(W_dims, "vectors")
		s_dW = initialize_zeros(W_dims, "matrices")
		s_db = initialize_zeros(W_dims, "vectors")

		return new(
			A,
			Z,
			dA,
			dZ,
			dW,
			db,
			v_dW,
			v_db,
			s_dW,
			s_db,
		)
	end
end


mutable struct HyperParameters
	learning_rate   ::Float32
	mini_batch_size ::Int
	optimization    ::String
	λ               ::Float32
	# Momentum optimization parameters
	β               ::Float32
	# Adam optimization parameters
	β₁              ::Float32
	β₂              ::Float32
	ϵ               ::Float32
	t               ::Int

	function HyperParameters(hparams_dict)

		return new(
			Float32(get(hparams_dict, "learning_rate", 0.01)),
			get(hparams_dict, "mini_batch_size", 64),
			get(hparams_dict, "optimization", "gd"),
			Float32(get(hparams_dict, "λ", 0.1)),
			Float32(get(hparams_dict, "β", 0.9)),
			Float32(get(hparams_dict, "β₁", 0.9)),
			Float32(get(hparams_dict, "β₂", 0.999)),
			Float32(get(hparams_dict, "ϵ", 1e-8)),
			get(hparams_dict, "t", 0),
		)
	end
end


mutable struct MiniBatch
	X::Array{Float32, 2}
	Y::Array{Float32, 2}
end


mutable struct NeuralNetwork
	X       ::Array{Float32, 2}
	Y       ::Array{Float32, 2}
	L       ::Int
	params  ::Parameters
	hparams ::HyperParameters
	cache   ::Cache
	rng     ::MersenneTwister
	J       ::Array{Float32, 1}
	epoch   ::Int

	function NeuralNetwork(
		X            ::Array,
		Y            ::Array;
		params_dict  ::Dict=Dict(),
		hparams_dict ::Dict=Dict(),
	)

	# Explicitly typecast to Float32 to ensure all calcs use the same precision
	X = Float32.(X)
	Y = Float32.(Y)

	# Explicitly typecast to Any dicts to avoid type errors
	params_dict  = convert(Dict{Any, Any}, params_dict)
	hparams_dict = convert(Dict{Any, Any}, hparams_dict)

	# Set values for useful variables
	params_dict["input_layer_size"] = size(X, 1)
	params_dict["layer_sizes"]      = get(params_dict, "layer_sizes", [3, 1])

	# Create the rng and seed it with the value from params_dict
	rng                = MersenneTwister(get(params_dict, "seed", 8))
	params_dict["rng"] = rng

	# Initialize instances of custom types
	params  = Parameters(params_dict)
	hparams = HyperParameters(hparams_dict)
	cache   = Cache(params_dict)

	# Set starting values of variables
	L = length(params_dict["layer_sizes"])
	J = []
	epoch = 0

	NN = new(
		X,
		Y,
		L,
		params,
		hparams,
		cache,
		rng,
		J,
		epoch,
	)
	end
end

