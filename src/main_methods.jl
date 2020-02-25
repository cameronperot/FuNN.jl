function train!(NN::NeuralNetwork, n_epochs::Int)
	for epoch in 1:n_epochs
		NN.epoch += 1
		push!(NN.J, 0)

		mini_batches = random_mini_batches(NN)
		for mini_batch in mini_batches
			propagate_forward!(NN, mini_batch)
			compute_cost!(NN, mini_batch)
			propagate_back!(NN, mini_batch)
			update_params!(NN)
		end
	end
end


function predict(params::Parameters, X::Array{T, 2}) where {T <: AbstractFloat}
	X  = eltype(params.W[1]).(X)
	gs = params.activation_functions
	L  = length(params.layer_sizes)

	A = g[gs[1]](params.W[1] * X .+ params.b[1])
	for l in 2:L
		A = g[gs[l]](params.W[l] * A .+ params.b[l])
	end

	return A
end


function predict(NN::NeuralNetwork, X::Array{T, 2}) where {T <: AbstractFloat}
	predict(NN.params, X)
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
