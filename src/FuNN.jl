module FuNN

using Random

export

# Classifier
NeuralNetwork,

# Main methods
train!,
predict,

# Activation and loss functions
relu,
logistic,
softmax,
cross_entropy_binary,
cross_entropy_multi,
mean_squared_error

include("./NeuralNetwork.jl")
include("./back_propagation.jl")
include("./forward_propagation.jl")
include("./main_methods.jl")
include("./update_methods.jl")

end # module
