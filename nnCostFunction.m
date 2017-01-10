function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

E = eye(num_labels);
Y = zeros(m,num_labels);
for i=1:m
  if y(i) == 0 % Because numeration in matlab begin from 1
    Y(i,:) = E(10,:); 
  else
    Y(i,:) = E(y(i),:);
  end
end

% ==== Feedforward ====
h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
hypothesis = h2;

cost = sum(-Y .* log(hypothesis) - (1-Y) .* log(1 - hypothesis));


new_theta1 = Theta1(:,2:end);
reg1 = sum(sum(new_theta1.^2));
new_theta2 = Theta2(:,2:end);
reg2 = sum(sum(new_theta2.^2));
regularization = lambda/(2*m) * (reg1+reg2);
J = 1/m * sum(cost) + regularization;

% ==== Backpropagation ==== 

big_delta1 = zeros(hidden_layer_size,input_layer_size+1);
big_delta2 = zeros(num_labels, hidden_layer_size + 1);

X = [ones(m,1) X];

for i=1:m
  a1 = X(i,:);
  z2 = Theta1 * a1';
  a2 = sigmoid(z2);
  a2 = [1;a2];
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);
  
  delta3 = a3 - Y(i,:)';
  delta2 = Theta2(:,2:end)' * delta3.*sigmoidDerivative(z2);
 
  big_delta1 = big_delta1 + delta2 * a1;
  big_delta2 = big_delta2 + delta3 * a2';  
end

Theta1_grad(:,1) = 1/m * big_delta1(:,1);
Theta1_grad(:,2:end) = 1/m * big_delta1(:,2:end) + lambda/m * Theta1(:,2:end);
Theta2_grad(:,1) = 1/m * big_delta2(:,1);
Theta2_grad(:,2:end) = 1/m * big_delta2(:,2:end) + lambda/m * Theta2(:,2:end);

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
