function g = sigmoidDerivative(z)

g = zeros(size(z));
sigm = sigmoid(z);
g = sigm.*(1-sigm);

end
