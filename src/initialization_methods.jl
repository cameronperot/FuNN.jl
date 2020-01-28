function initialize_params!(params_dict)
	layer_sizes      = params_dict["layer_sizes"]
	input_layer_size = params_dict["input_layer_size"]
	rng              = params_dict["rng"]

	L = length(layer_sizes)
	W = Array{Array{Float32, 2}, 1}(undef, L)
	b = Array{Array{Float32, 2}, 1}(undef, L)
	for l in 1:L
		if l ≠ 1
			W[l] = randn(rng, layer_sizes[l], layer_sizes[l-1]) .* sqrt(2 / layer_sizes[l-1])
		else
			W[l] = randn(rng, layer_sizes[l], input_layer_size) .* sqrt(2 / input_layer_size)
		end

		b[l] = zeros(layer_sizes[l], 1)
	end

	params_dict["W"] = W
	params_dict["b"] = b
	params_dict["activation_functions"] = get(
		params_dict,
		"activation_functions",
		push!(["relu" for l in 1:L-1], "logistic")
	)
end


function random_mini_batches(NN::NeuralNetwork)
	m = size(NN.X, 2)
	K = ceil(m / NN.hparams.mini_batch_size)
	permutation = randperm(NN.rng, m)
	mini_batches = MiniBatch[]

	for k in 1:K-1
		a = Int((k - 1) * NN.hparams.mini_batch_size + 1)
		b = Int(k * NN.hparams.mini_batch_size)
		push!(mini_batches, MiniBatch(NN.X[:, permutation][:, a:b], NN.Y[:, permutation][:, a:b]))
	end

	a = Int((K - 1) * NN.hparams.mini_batch_size + 1)
	push!(mini_batches, MiniBatch(NN.X[:, permutation][:, a:end], NN.Y[:, permutation][:, a:end]))

	return mini_batches
end
