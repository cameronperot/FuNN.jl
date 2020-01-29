function relu(Z::Array{Float32, 2})
	return @. Z * (Z > 0)
end


function logistic(Z::Array{Float32, 2})
	return @. 1 / (1 + exp(-Z))
end


function softmax(Z::Array{Float32, 2})
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


function cross_entropy_binary(Y::Array{Float32, 2}, Y_hat::Array{Float32, 2}, m::Float32)
 	return -sum(Y .* log.(Y_hat) .+ (1 .- Y) .* log.(1 .- Y_hat)) / m
end


function cross_entropy_multi(Y::Array{Float32, 2}, Y_hat::Array{Float32, 2}, m::Float32)
	return -sum(Y .* log.(Y_hat)) / m
end


function mean_squared_error(Y::Array{Float32, 2}, Y_hat::Array{Float32, 2}, m::Float32)
	return sum((Y .- Y_hat).^2) / 2 / m
end


L = Dict{String, Function}(
	"cross_entropy_binary" => cross_entropy_binary,
	"cross_entropy_multi"  => cross_entropy_multi,
	"mean_squared_error"   => mean_squared_error,
)


function compute_cost!(NN::NeuralNetwork, mini_batch::MiniBatch)
	m = Float32(size(NN.X, 2))
	NN.J[NN.epoch] += L[NN.params.loss_function](mini_batch.Y, NN.cache.A[NN.L], m)

	# Add in regularization if using gradient descent
	if NN.hparams.optimization == "gd"
		regularization = Float32(0)
		for l in 1:NN.L
			regularization += sum(NN.params.W[l].^2)
		end
		NN.J[NN.epoch] += regularization * NN.hparams.λ / 2 / m
	end
end
