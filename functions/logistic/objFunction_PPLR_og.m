function J = objFunction_PPLR_og(theta, X, y, epsilon)



m = length(y); % number of training examples

lambda = 0.1;

t = gamrnd(length(theta),2/epsilon);

b = rand( length(theta),1 );

s = sign(rand(length(theta),1)-0.5);

b = s.*b/norm(b)*t;

h = sigmoid(X * theta);

J = 1/2*lambda*theta'*theta + b'*theta...
    -    sum( (y .* log(h+eps)) + ((1 - y) .* log(1 - h+eps)) );




end