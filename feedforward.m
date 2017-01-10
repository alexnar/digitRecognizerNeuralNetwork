% return:
% cost function J
% predicted output for each training example Y
function [J Y] = feedforward(X, y, Theta1, Theta2, lambda, output_layer_size)

% E: 
% 1 0 ... 0 0 - E(1) is equal to 1
% 0 1 ... 0 0 - E(2) is equal to 2
%     ...
% 0 0 ... 1 0 - E(9) is equal to 9
% 0 0 ... 0 1 - E(10) is equal to 0 ( because numeration in matlab begins from 1)

E = eye(output_layer_size);
m = size(X, 1);
Y = zeros(m,output_layer_size);

for i=1:m
  if y(i) == 0
    Y(i,:) = E(10,:); 
  else
    Y(i,:) = E(y(i),:);
  end
end

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

end