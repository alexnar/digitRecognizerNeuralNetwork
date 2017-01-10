function [Theta1 Theta2 cost] = trainNN(X, y, lambda, input_layer_size, hidden_layer_size, output_layer_size)

% === Random initialization all Theta weights ===

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, output_layer_size);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% checkNNGradients(1);


% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   output_layer_size, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
options = optimset('MaxIter', 50);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 output_layer_size, (hidden_layer_size + 1));
                 
end