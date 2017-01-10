function execute = execute(X, y)

% X - input features
% y - output variable
X = (X - mean(X(:)))/(max(X(:)) - min(X(:)));
size(X)
input_layer_size = 784; % 28*28
hidden_layer_size = 40;
output_layer_size = 10;

lambda = 0.2

[Theta1 Theta2] = trainNN(X, y, lambda, input_layer_size, hidden_layer_size, output_layer_size);

                 
save('weights.mat','Theta1','Theta2');

pred = predict(Theta1, Theta2, X);
pred(pred==10)=0;
size(pred)
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

end