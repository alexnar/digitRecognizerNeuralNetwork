function [error_train, error_cv] = trainExamples(X, y, Xcv, ycv, lambda, ...
                                                  input_layer_size, ...
                                                  hidden_layer_size, ...
                                                  output_layer_size)

m = size(X, 1);
error_train = zeros(m, 1);
error_cv   = zeros(m, 1);

X = featureScaling(X);
Xcv = featureScaling(Xcv);

for i = 1:1200:m
  Xtrain = X(1:i,:);
  ytrain = y(1:i);
  [Theta1 Theta2] = trainNN(Xtrain,ytrain,lambda, ...
                            input_layer_size, hidden_layer_size, output_layer_size);
  [Jtrain Y] = feedforward(Xtrain, ytrain, Theta1, Theta2, 0, output_layer_size);
  [Jcv Y] = feedforward(Xcv, ycv, Theta1, Theta2, 0, output_layer_size);
  Jtrain
  Jcv
  error_train(i) = Jtrain;
  error_cv(i) = Jcv;
end

end
