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

	function Cache(params)
		input_layer_size = params["input_layer_size"]
		layer_sizes = params["layer_sizes"]
		W_dims = pushfirst!(copy(layer_sizes), input_layer_size)

		A = [zeros(Float32, layer_sizes[l], 1) for l in 1:length(layer_sizes)]
		Z = [zeros(Float32, layer_sizes[l], 1) for l in 1:length(layer_sizes)]
		dA = [zeros(Float32, layer_sizes[l], 1) for l in 1:length(layer_sizes)]
		dZ = [zeros(Float32, layer_sizes[l], 1) for l in 1:length(layer_sizes)]
		dW = [zeros(Float32, W_dims[l+1], W_dims[l]) for l in 1:length(layer_sizes)]
		db = [zeros(Float32, W_dims[l+1], 1) for l in 1:length(layer_sizes)]

		# Adam and momentum optimization cache
		v_dW = [zeros(Float32, W_dims[l+1], W_dims[l]) for l in 1:length(layer_sizes)]
		v_db = [zeros(Float32, W_dims[l+1], 1) for l in 1:length(layer_sizes)]
		s_dW = [zeros(Float32, W_dims[l+1], W_dims[l]) for l in 1:length(layer_sizes)]
		s_db = [zeros(Float32, W_dims[l+1], 1) for l in 1:length(layer_sizes)]

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
			get(hparams_dict, "λ", 0.1),
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

	X = Float32.(X)
	Y = Float32.(Y)
	params_dict = convert(Dict{Any, Any}, params_dict)
	hparams_dict = convert(Dict{Any, Any}, hparams_dict)
	params_dict["input_layer_size"] = size(X, 1)
	params_dict["layer_sizes"] = get(params_dict, "layer_sizes", [3, 1])
	rng = MersenneTwister(get(params_dict, "seed", 8))
	params_dict["rng"] = rng

	params = Parameters(params_dict)
	hparams = HyperParameters(hparams_dict)
	cache = Cache(params_dict)

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

