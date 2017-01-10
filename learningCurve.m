function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)


for i = 1:m
  % Compute train/cross validation errors using training examples
  Xtrain = X(1:i,:);
  ytrain = y(1:i);
  
  [error_train(i), grad] = linearRegCostFunction(Xtrain,ytrain,theta,0);
  [error_val(i), grad] = linearRegCostFunction(Xval,yval,theta,0);
  

end

















% -------------------------------------------------------------

% =========================================================================

end
