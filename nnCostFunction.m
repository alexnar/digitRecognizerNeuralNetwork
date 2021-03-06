function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   output_layer_size, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 output_layer_size, (hidden_layer_size + 1));

m = size(X, 1);         
J = 0;

[J Y] = feedforward(X, y, Theta1, Theta2, lambda, output_layer_size);

grad = backpropagation(X, Theta1, Theta2, Y, lambda, ...
               input_layer_size, hidden_layer_size, output_layer_size);


end
