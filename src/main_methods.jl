function train!(NN::NeuralNetwork, n_epochs::Int)
	for epoch in 1:n_epochs
		NN.epoch += 1
		mini_batches = random_mini_batches(NN)
		push!(NN.J, Float32(0))
		for mini_batch in mini_batches
			propagate_forward!(NN, mini_batch)
			compute_cost!(NN, mini_batch)
			propagate_back!(NN, mini_batch)
			update_params!(NN)
		end
	end
end


function predict(params::Parameters, X::Array)
	X = Float32.(X)
	gs = NN.params.activation_functions
	L = length(params.layer_sizes)
	A = g[gs[1]](params.W[1] * X .+ NN.params.b[1])
	for l in 2:L-1
		A = g[gs[l]](params.W[l] * A .+ NN.params.b[l])
	end
	return g[gs[L]](params.W[L] * A .+ NN.params.b[L]) .> 0.5
end


function predict(NN::NeuralNetwork, X::Array)
	predict(NN.params, X)
end
