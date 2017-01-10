function Weights = randInitializeWeights(incoming_layer_size, outgoing_layer_size)

Weights = zeros(outgoing_layer_size, 1 + incoming_layer_size);
epsilon_init = 0.12;
Weights = rand(outgoing_layer_size, 1 + incoming_layer_size) * 2 * epsilon_init - epsilon_init;


end
