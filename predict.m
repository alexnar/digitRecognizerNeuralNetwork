function p = predict(Theta1, Theta2, X)
% Predict the label of trained neural network by given input

X = featureScaling(X);

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

X = [ones(m,1) X];
for i=1:m
  a1 = X(i,:);
  z2 = Theta1 * a1';
  a2 = sigmoid(z2);
  a2 = [1;a2];
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);
  [val index] = max(a3);
  p(i) = index;
end
%h1 = sigmoid([ones(m, 1) X] * Theta1');
%h2 = sigmoid([ones(m, 1) h1] * Theta2');
%[dummy, p] = max(h2, [], 2);


end
