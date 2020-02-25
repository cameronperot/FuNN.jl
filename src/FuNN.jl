module FuNN

using Random

export

# Classifier
NeuralNetwork,

# Main methods
train!,
predict

include("./NeuralNetwork.jl")
include("./back_propagation.jl")
include("./forward_propagation.jl")
include("./main_methods.jl")
include("./update_methods.jl")

end # module
