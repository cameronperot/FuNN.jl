function gradient_descent!(NN::NeuralNetwork)
	for l in 1:NN.L
		NN.params.W[l] .-= NN.hparams.learning_rate .* NN.cache.dW[l]
		NN.params.b[l] .-= NN.hparams.learning_rate .* NN.cache.db[l]
	end
end


function gradient_descent_momentum!(NN::NeuralNetwork)
	β = NN.hparams.β
	for l in 1:NN.L
		NN.cache.v_dW[l] = β .* NN.cache.v_dW[l] .+ (1 - β) .* NN.cache.dW[l]
		NN.cache.v_db[l] = β .* NN.cache.v_db[l] .+ (1 - β) .* NN.cache.db[l]
		NN.params.W[l] .-= NN.hparams.learning_rate .* NN.cache.v_dW[l]
		NN.params.b[l] .-= NN.hparams.learning_rate .* NN.cache.v_db[l]
	end
end


function gradient_descent_adam!(NN::NeuralNetwork)
	NN.hparams.t += 1
	α  = NN.hparams.learning_rate
	β₁ = NN.hparams.β₁
	β₂ = NN.hparams.β₂
	ϵ  = NN.hparams.ϵ
	t  = NN.hparams.t
	for l in 1:NN.L
		NN.cache.v_dW[l] = β₁ .* NN.cache.v_dW[l] .+ (1 - β₁) .* NN.cache.dW[l]
		NN.cache.v_db[l] = β₁ .* NN.cache.v_db[l] .+ (1 - β₁) .* NN.cache.db[l]

		NN.cache.s_dW[l] = β₂ .* NN.cache.s_dW[l] .+ (1 - β₂) .* NN.cache.dW[l].^2
		NN.cache.s_db[l] = β₂ .* NN.cache.s_db[l] .+ (1 - β₂) .* NN.cache.db[l].^2

		NN.params.W[l] -= (α / (1 - β₁^t)) .* NN.cache.v_dW[l] ./ (sqrt.(NN.cache.s_dW[l] ./ (1 - β₂^t)) .+ ϵ)
		NN.params.b[l] -= (α / (1 - β₁^t)) .* NN.cache.v_db[l] ./ (sqrt.(NN.cache.s_db[l] ./ (1 - β₂^t)) .+ ϵ)
	end
end


function update_params!(NN::NeuralNetwork)
	if NN.hparams.optimization == "gd"
		gradient_descent!(NN)
	elseif NN.hparams.optimization == "momentum"
		gradient_descent_momentum!(NN)
	elseif NN.hparams.optimization == "adam"
		gradient_descent_adam!(NN)
	end
end
