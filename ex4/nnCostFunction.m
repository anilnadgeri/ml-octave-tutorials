function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% m*10
y_binary = zeros(m, num_labels);
 
%convert y into y_binary
%for i= 1:m
%  y_binary(i, y(i)) = 1;
%end
ind = sub2ind( size(y_binary), 1:rows(y_binary), y');
y_binary(ind) = 1;
 
%calculate h(theta)
 
%add theta1 column
a1 = [ones(m,1), X];
 
% a1 = m*401
% Theta1 = 25*401
% Theta2 = 10*26
a2 = sigmoid(a1*Theta1'); % m*25
 
%add theta1 column
a2 = [ones(m,1) a2]; % m*26
 
h_theta = sigmoid(a2*Theta2'); % m*10
 
% regularized cost. For regularised expression, remove first column from Theta1 and Theta2
J = sum( (-y_binary.*log(h_theta) - (1-y_binary).*log(1-h_theta))(:) )/m + lambda*(sum((Theta1(:,2:end).^2)(:)) + sum((Theta2(:,2:end).^2)(:)))/(2*m);


% iterate through all training examples and accumulate gradients w.r.t. all thetas
 
for t = 1 : m
               % t_th training set
               a1_t = a1(t,:)'; % 401*1
               %fprintf('Size of a1_t %d %d\n', size(a1_t,1), size(a1_t,2));
              
               % transpose y_t
               y_t = y_binary(t,:)'; % 10*1
               %fprintf('Size of y_t %d %d\n', size(y_t,1), size(y_t,2));
              
               % activation for layer 2
               % 25*1 =       25*401 401*1
               a2_t = sigmoid(Theta1*a1_t);
              
               % add theta0
               %fprintf('Size of a2_t %d %d\n', size(a2_t,1), size(a2_t,2));
               a2_t = [1; a2_t]; % 26*1
              
               % activation for layer 3
               % 10*1 =       10*26  26*1
               a3_t = sigmoid(Theta2*a2_t);
              
               % delta at layer 3
               % 10*1
               d3 = a3_t - y_t;
              
               % delta at layer 2
               % 26*1=(26*10  10*1)    26*1    26*1
               d2 =   (Theta2'*d3).* (a2_t.*(1-a2_t)); 
               % skip d2_0
               % 25*1
               d2 = d2(2:end);  
 
              
               % accumulate Theta2 gradients
               % 10*26     =                10*1   1*26
               Theta2_grad = Theta2_grad + ( d3  * a2_t' );
              
               % accumulate Theta1 gradients
               % 25*401    =                25*1 1*401
               Theta1_grad = Theta1_grad + ( d2 * a1_t' );
              
end
 
% Ignore theta_0 i.e. first column in Theta2 for regularization expression
Theta2_reg = Theta2;
Theta2_reg(:,1) = 0;
 
% average the gradients for Theta2
% 10*26
Theta2_grad = Theta2_grad/m + (lambda/m)*(Theta2_reg);
 
% Ignore theta_0 i.e. first column in Theta1 for regularization expression
Theta1_reg = Theta1;
Theta1_reg(:,1) = 0;
 
% average the gradients for Theta1
% 25*401
Theta1_grad = Theta1_grad/m + (lambda/m)*(Theta1_reg);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
