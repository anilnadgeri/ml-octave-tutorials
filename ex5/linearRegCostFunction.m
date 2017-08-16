function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% theta = n*1
% X = m*n
% y = m*1  
 
% h_theta = m*1
h_theta = X*theta;
 
% cost function
J = sum((h_theta - y).^2)/(2*m) + lambda*sum(theta(2:end).^2)/(2*m);
 
% gradients
theta_grad = theta;
theta_grad(1) = 0;
 
% n*1= (n*m)(m*1)             (n*1)
grad = (X'*(h_theta - y))/m + theta_grad*lambda/m;


% =========================================================================

grad = grad(:);

end
