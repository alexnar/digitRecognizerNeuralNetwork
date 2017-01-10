function grad = backpropagation(X, Theta1, Theta2, Y, lambda, ...
                        input_layer_size, ... 
                        hidden_layer_size, ...
                        output_layer_size)

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

big_delta1 = zeros(hidden_layer_size,input_layer_size+1);
big_delta2 = zeros(output_layer_size, hidden_layer_size + 1);

m = size(X, 1);
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