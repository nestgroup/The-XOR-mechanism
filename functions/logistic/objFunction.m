function [J ] = objFunction(theta, X, y)



m = length(y); % number of training examples


h = sigmoid(X * theta);
J = -   sum( (y .* log(h+eps)) + ((1 - y) .* log(1 - h+eps)) );


end