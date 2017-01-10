function [lambda_vec, error_train, error_cv] = ...
    validationCurve(X, y, Xcv, ycv, input_layer_size, hidden_layer_size, output_layer_size)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.07 0.1 0.2 0.3 0.4 0.5 0.7 1 2 3 4 5]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_cv = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%

for i = 1:length(lambda_vec)
  lambda = lambda_vec(i);
  % Compute train / val errors when training linear 
  [Theta1 Theta2 cost] = trainNN(X,y,lambda, input_layer_size, hidden_layer_size, output_layer_size);
  [Jtrain Y] = feedforward(X, y, Theta1, Theta2, lambda, output_layer_size);
  [Jcv Y] = feedforward(Xcv, ycv, Theta1, Theta2, lambda, output_layer_size);
  error_train(i) = Jtrain;
  error_cv(i) = Jcv;
end

save('errors.mat', 'error_train', 'error_cv');










% =========================================================================

end
