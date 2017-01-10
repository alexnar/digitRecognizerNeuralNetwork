function [error_train, error_cv] = learningCurve(X, y, Xcv, ycv, lambda, ...
                                                  input_layer_size, ...
                                                  hidden_layer_size, ...
                                                  output_layer_size)

% m = size(X, 1);
m = 100;
error_train = zeros(m, 1);
error_cv   = zeros(m, 1);

for i = 1:m
  % Compute train/cross validation errors using training examples
  Xtrain = X(1:i,:);
  ytrain = y(1:i);
  [Theta1 Theta2 cost] = trainNN(Xtrain,ytrain,lambda, input_layer_size, hidden_layer_size, output_layer_size);
  [Jtrain Y] = feedforward(Xtrain, ytrain, Theta1, Theta2, lambda, output_layer_size);
  [Jcv Y] = feedforward(Xcv, ycv, Theta1, Theta2, lambda, output_layer_size);
  error_train(i) = Jtrain;
  error_cv(i) = Jcv;

end



end
