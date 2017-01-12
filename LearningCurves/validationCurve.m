function [lambda_vec, error_train, error_cv] = ...
    validationCurve(X, y, Xcv, ycv, input_layer_size, hidden_layer_size, output_layer_size)
    
% Compute the train and validation errors needed for different lambda

X = X/255;
Xcv = Xcv/255;

lambda_vec = [0 0.01 0.1 0.5 0.7 0.9 1 5 10]';

error_train = zeros(length(lambda_vec), 1);
error_cv = zeros(length(lambda_vec), 1);


for i = 1:length(lambda_vec)
  lambda = lambda_vec(i)
  % Compute train / val errors when training linear 
  [Theta1 Theta2] = trainNN(X,y,lambda, input_layer_size, ...
                                 hidden_layer_size, output_layer_size);
  [Jtrain Y] = feedforward(X, y, Theta1, Theta2, 0, output_layer_size);
  [Jcv Y] = feedforward(Xcv, ycv, Theta1, Theta2, 0, output_layer_size);
  error_train(i) = Jtrain;
  error_cv(i) = Jcv;
end

save('errors.mat', 'error_train', 'error_cv');


end
