@. function relu(Z::Array{Float32, 2})
	return Z * (Z > 0)
end


@. function logistic(Z::Array{Float32, 2})
	return 1 / (1 + exp(-Z))
end


g = Dict{String, Function}("relu" => relu, "tanh" => tanh, "logistic" => logistic);


function propagate_forward!(NN::NeuralNetwork, mini_batch::MiniBatch)
	gs = NN.params.activation_functions
	NN.cache.Z[1] = NN.params.W[1] * mini_batch.X .+ NN.params.b[1]
	NN.cache.A[1] = g[gs[1]](NN.cache.Z[1])
	for l in 2:NN.L-1
		NN.cache.Z[l] = NN.params.W[l] * NN.cache.A[l-1] .+ NN.params.b[l]
		NN.cache.A[l] = g[gs[l]](NN.cache.Z[l])
	end
	NN.cache.Z[NN.L] = NN.params.W[NN.L] * NN.cache.A[NN.L-1] .+ NN.params.b[NN.L]
	NN.cache.A[NN.L] = g[gs[NN.L]](NN.cache.Z[NN.L])
end


function cross_entropy(Y::Array{Float32, 2}, Y_hat::Array{Float32, 2}, m)
 	return -sum(Y .* log.(Y_hat) .+ (1 .- Y) .* log.(1 .- Y_hat)) / m
end


function compute_cost!(NN::NeuralNetwork, mini_batch::MiniBatch)
	NN.J[NN.epoch] += cross_entropy(mini_batch.Y, NN.cache.A[NN.L], size(NN.X, 2))
end
