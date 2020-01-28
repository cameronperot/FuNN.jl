module FuNN

using Random

export

# Classifier
NeuralNetwork,

# Main methods
train!,
predict

include("./back_propagation.jl")
include("./Classifier.jl")
include("./forward_propagation.jl")
include("./initialization_methods.jl")
include("./main_methods.jl")
include("./update_methods.jl")

end # module
