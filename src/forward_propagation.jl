function relu(Z::Array{T, 2}) where {T <: AbstractFloat}
	return @. Z * (Z > 0)
end


function logistic(Z::Array{T, 2}) where {T <: AbstractFloat}
	return @. 1 / (1 + exp(-Z))
end


function softmax(Z::Array{T, 2}) where {T <: AbstractFloat}
	e = exp.(Z)
	return e ./ sum(e, dims=1)
end


g = Dict{String, Function}(
	"relu"     => relu,
	"tanh"     => tanh,
	"logistic" => logistic,
	"softmax"  => softmax,
	)


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


function cross_entropy_binary(Y::Array{T, 2}, Y_hat::Array{T, 2}) where {T <: AbstractFloat}
 	return -sum(Y .* log.(Y_hat) .+ (1 .- Y) .* log.(1 .- Y_hat))
end


function cross_entropy_multi(Y::Array{T, 2}, Y_hat::Array{T, 2}) where {T <: AbstractFloat}
	return -sum(Y .* log.(Y_hat))
end


function mean_squared_error(Y::Array{T, 2}, Y_hat::Array{T, 2}) where {T <: AbstractFloat}
	return sum((Y .- Y_hat).^2) / 2
end


L = Dict{String, Function}(
	"cross_entropy_binary" => cross_entropy_binary,
	"cross_entropy_multi"  => cross_entropy_multi,
	"mean_squared_error"   => mean_squared_error,
	)


function compute_cost!(NN::NeuralNetwork, mini_batch::MiniBatch)
	NN.J[NN.epoch] += L[NN.params.loss_function](mini_batch.Y, NN.cache.A[NN.L]) / NN.m

	# Add in regularization if using gradient descent
	if NN.hparams.optimization == "gd"
		regularization = typeof(NN.m)(0)
		for l in 1:NN.L
			regularization += sum(NN.params.W[l].^2)
		end
		NN.J[NN.epoch] += regularization * NN.hparams.λ / 2 / NN.m
	end
end
