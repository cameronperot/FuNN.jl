function drelu_dZ(NN::NeuralNetwork, l::Int)
	return @. NN.cache.Z[l] > 0
end


function dlogistic_dZ(NN::NeuralNetwork, l::Int)
	return @. NN.cache.A[l] * (1 - NN.cache.A[l])
end


function dtanh_dZ(NN::NeuralNetwork, l::Int)
	return @. 1 - NN.cache.A[l]^2
end


function dsoftmax_dZ(NN::NeuralNetwork, l::Int)
	return @. NN.cache.A[l] * (1 - NN.cache.A[l])
end


function propagate_back!(NN::NeuralNetwork, mini_batch::MiniBatch)
	m = Float32(size(mini_batch.X, 2))
	for l in NN.L:-1:1
		compute_dJ_dA!(NN, l, mini_batch)
		compute_dJ_dZ!(NN, l)
		compute_dJ_dW!(NN, l, mini_batch, m)
		compute_dJ_db!(NN, l, m)
	end
end


function compute_dJ_dA!(NN::NeuralNetwork, l::Int, mini_batch::MiniBatch)
	if l ≠ NN.L
		NN.cache.dA[l] = NN.params.W[l+1]' * NN.cache.dZ[l+1]
	else
		if NN.params.loss_function == "cross_entropy_binary"
			NN.cache.dA[l] = @. -mini_batch.Y / NN.cache.A[NN.L] + (1 - mini_batch.Y) / (1 - NN.cache.A[NN.L])
		elseif NN.params.loss_function == "cross_entropy_multi"
			NN.cache.dA[l] = @. -mini_batch.Y / NN.cache.A[NN.L]
		elseif NN.params.loss_function == "mean_squared_error"
			NN.cache.dA[l] = @. NN.cache.A[NN.L] - mini_batch.Y
		else
			error("Unknown loss function provided in params")
		end
	end
end


function compute_dJ_dZ!(NN::NeuralNetwork, l::Int)
	if NN.params.activation_functions[l] == "tanh"
		NN.cache.dZ[l] = NN.cache.dA[l] .* drelu_dZ(NN, l)
	elseif NN.params.activation_functions[l] == "relu"
		NN.cache.dZ[l] = NN.cache.dA[l] .* dtanh_dZ(NN, l)
	elseif NN.params.activation_functions[l] == "logistic"
		NN.cache.dZ[l] = NN.cache.dA[l] .* dlogistic_dZ(NN, l)
	elseif NN.params.activation_functions[l] == "softmax"
		NN.cache.dZ[l] = NN.cache.dA[l] .* dsoftmax_dZ(NN, l)
	else
		error("Unknown activation function(s) provided in params")
	end
end


function compute_dJ_dW!(NN::NeuralNetwork, l::Int, mini_batch::MiniBatch, m::Float32)
	if l ≠ 1
		NN.cache.dW[l] = (NN.cache.dZ[l] ./ m) * NN.cache.A[l-1]'
	else
		NN.cache.dW[l] = (NN.cache.dZ[l] ./ m) * mini_batch.X'
	end

	# Add in regularization if using gradient descent
	if NN.hparams.optimization == "gd"
		NN.cache.dW[l] += (NN.hparams.λ / m) .* NN.params.W[l]
	end
end


function compute_dJ_db!(NN::NeuralNetwork, l::Int, m::Float32)
	NN.cache.db[l] = sum(NN.cache.dZ[l], dims=2) ./ m
end
