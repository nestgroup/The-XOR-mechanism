function [J ] = objFunction_PPLR(theta, X, y,b)



m = length(y); % number of training examples


h = sigmoid(X * theta);
J = -  1/m* sum( (y .* log(h+eps)) + ((1 - y) .* log(1 - h+eps)) ) + 1/m*  sum (b.*theta);


end