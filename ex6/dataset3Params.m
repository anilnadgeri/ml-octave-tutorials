function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

c_values = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_values = [0.01 0.03 0.1 0.3 1 3 10 30];

% inititialize 2D matrix to catpure errors
prediction_errors = zeros(length(c_values), length(sigma_values));

c_index = 0;
sigma_index = 0;

for c_value = c_values
  c_index += 1;
  
  for sigma_value = sigma_values
    sigma_index += 1;
    
    model = svmTrain(X, y, c_value, @(x1, x2)gaussianKernel(x1, x2, sigma_value));
    predictions = svmPredict(model, Xval);
    prediction_errors(c_index, sigma_index) = sum(predictions != yval);
  
  end
  
  sigma_index = 0;

end

[minval, c_index] = min(min(prediction_errors, [], 2));
[minval, sigma_index] = min(min(prediction_errors, [], 1));

C = c_values(c_index);
sigma = sigma_values(sigma_index);

% =========================================================================

end
