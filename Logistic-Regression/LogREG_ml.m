% Microchips classification --------------------------------

% Initialization
clear; close all; clc;

% Load the data
fprintf('Loading the data...\n\n');

data = load('microchips_tests.csv');

X = data(:, 1:2);
y = data(:, 3);

% Plotting the data
fprintf('Plotting the data...\n\n');

% Find indices of positive and negative examples.
positiveIndices = find(y == 1);
negativeIndices = find(y == 0);

% Plot examples.
hold on;
plot(X(positiveIndices, 1), X(positiveIndices, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(negativeIndices, 1), X(negativeIndices, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

% Draw labels and Legend
xlabel('Microchip Test 1');
ylabel('Microchip Test 2');
legend('y = 1', 'y = 0');

% Running logistic regression ------------------------------------------
fprintf('Running logistic regression...\n\n');

% Add more polynomial features in order to make decision boundary to have more complex curve form.
polynomial_degree = 6;
X = add_polynomial_features(X(:, 1), X(:, 2), polynomial_degree);

% Run the regression.
lambda = 1;
[theta, J, J_history, exit_flag] = logistic_regression_train(X, y, lambda);

fprintf('- Initial cost: %f\n', J_history(1));
fprintf('- Optimized cost: %f\n\n', J);

% Plotting decision boundaries
fprintf('Plotting decision boundaries...\n\n');

% Generate a grid range.
u = linspace(-1, 1, 50);
v = linspace(-1, 1, 50);
z = zeros(length(u), length(v));
% Evaluate z = (x * theta) over the grid.
for i = 1:length(u)
    for j = 1:length(v)
        % Add polinomials.
        x = add_polynomial_features(u(i), v(j), polynomial_degree);
        % Add ones.
        x = [ones(size(x, 1), 1), x];
        z(i, j) = x * theta;
    end
end

% It is mportant to transpose z before calling the contour.
z = z';

% Plot z = 0
% Notice you need to specify the range [0, 0]
contour(u, v, z, [0, 0], 'LineWidth', 2);
title(sprintf('lambda = %g', lambda));
legend('y = 1', 'y = 0', 'Decision boundary');

hold off;

% Trying to predict custom experiments 
fprintf('Trying to predict custom experiments...\n\n');

x = [
    0, 0;
    -0.5, -0.5
];

% Add polinomials.
x = add_polynomial_features(x(:, 1), x(:, 2), polynomial_degree);
% Add ones.
x = [ones(size(x, 1), 1), x];

probabilities = hypothesis(x, theta);
fprintf(' %f \n', probabilities);

fprintf('\Press enter to train logistic regression to recognize digits.\n');
pause;


% Handwritten digits classification ------------------------
clear; close all; clc;

% Load training data
fprintf('Loading training data...\n');
load('digits.mat');

% Plotting some training example
fprintf('Visualizing data...\n');

% Randomly select 100 data points to display
random_digits_indices = randperm(size(X, 1));
random_digits_indices = random_digits_indices(1:100);

display_data(X(random_digits_indices, :));

% Setup the parameters you will use for this part of the exercise
input_layer_size = 400;  % 20x20 input images of digits.
num_labels = 10; % 10 labels, from 1 to 10 (note that we have mapped "0" to label 10).

fprintf('Training One-vs-All Logistic Regression...\n')
lambda = 0.01;
num_iterations = 50;
[all_theta] = one_vs_all(X, y, num_labels, lambda, num_iterations);

fprintf('Predict for One-Vs-All...\n')
pred = one_vs_all_predict(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);



% LOGISTIC REGRESSION function -----------------------
% Calculate the optimal thetas for given training set and output values.
function [theta, J, J_history, exit_flag] = logistic_regression_train(X, y, lambda)
    % X - training set.
    % y - training output values.
    % lambda - regularization parameter.

    % Calculate the number of training examples.
    m = size(y, 1);

    % Calculate the number of features.
    n = size(X, 2);

    % Add a column of ones to X.
    X = [ones(m, 1), X];

    % Initialize model parameters.
    initial_theta = zeros(n + 1, 1);

    % Run gradient descent.
    [theta, J, exit_flag] = gradient_descent(X, y, initial_theta, lambda);

    % Record the history of changing J.
    J_history = zeros(1, 1);
    J_history(1) = cost_function(X, y, initial_theta, lambda);
    J_history(2) = cost_function(X, y, theta, lambda);
end


% HYPOTHESIS function ------------------------
% It predicts the output values y based on the input values X and model parameters.
function [predictions] = hypothesis(X, theta)
    % Input:
    % X - input features - (m x n) matrix.
    % theta - our model parameters - (n x 1) vector.
    %
    % Output:
    % predictions - output values that a calculated based on model parameters - (m x 1) vector.
    %
    % Where:
    % m - number of training examples,
    % n - number of features.

    predictions = sigmoid(X * theta);
end


% COST function -------------------------
% It shows how accurate our model is based on current model parameters.
function [cost] = cost_function(X, y, theta, lambda)
    % X - training set.
    % y - training output values.
    % theta - model parameters.
    % lambda - regularization parameter.

    % Initialize number of training examples.
    m = length(y);

    % Calculate hypothesis.
    predictions = hypothesis(X, theta);

    % Calculate regularization parameter.
    % Remmber that we should not regularize the parameter theta_zero.
    theta_cut = theta(2:end, 1);
    regularization_param = (lambda / (2 * m)) * (theta_cut' * theta_cut);

    % Calculate cost function.
    cost = (-1 / m) * (y' * log(predictions) + (1 - y)' * log(1 - predictions)) + regularization_param;
end


% SIGMOID function.
function g = sigmoid(z)
    g = 1 ./ (1 + exp(-z));
end


% GRADIENT DESCENT function.
% Iteratively optimizes theta model parameters
function [theta, J, exit_flag] = gradient_descent(X, y, theta, lambda)
    % X - training set.
    % y - training output values.
    % theta - model parameters.
    % lambda - regularization parameter.

    % Set Options
    options = optimset('GradObj', 'on', 'MaxIter', 400);

    % Optimize
    [theta, J, exit_flag] = fminunc(@(t)(gradient_callback(X, y, t, lambda)), theta, options);
end


% GRADIENT STEP function.
% It performs one step of gradient descent for theta parameters.
function [gradients] = gradient_step(X, y, theta, lambda)
    % X - training set.
    % y - training output values.
    % theta - model parameters.
    % lambda - regularization parameter.

    % Initialize number of training examples.
    m = length(y);

    % Initialize variables we need to return.
    gradients = zeros(size(theta));

    % Calculate hypothesis.
    predictions = hypothesis(X, theta);

    % Calculate regularization parameter.
    regularization_param = (lambda / m) * theta;

    % Calculate gradient steps.
    gradients = (1 / m) * (X' * (predictions - y)) + regularization_param;

    % We should NOT regularize the parameter theta_zero.
    gradients(1) = (1 / m) * (X(:, 1)' * (predictions - y));
end


% GRADIENT CALLBACK function.
% This function is used as a callback function for fminunc and it aggregates
% cost and gradient values.
function [cost, gradients] = gradient_callback(X, y, theta, lambda)
    % X - training set.
    % y - training output values.
    % theta - model parameters.
    % lambda - regularization parameter.

    % Calculate cost function.
    cost = cost_function(X, y, theta, lambda);

    % Do one gradient step.
    gradients = gradient_step(X, y, theta, lambda);
end


% Generates polinomyal features of certain degree.
% This function is used to extend training set features with new features to get
% more complex form/shape if decision boundaries.
function out = add_polynomial_features(X1, X2, degree)
    % Returns a new feature array with more features, comprising of
    % X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    out = ones(size(X1(:, 1)));
    for i = 1:degree
        for j = 0:i
            out(:, end + 1) = (X1 .^ (i - j)) .* (X2 .^ j);
        end
    end
end


% Trains multiple logistic regression classifiers and returns all
% the classifiers in a matrix all_theta, where the i-th row of all_theta
% corresponds to the classifier for label i.
function [all_theta] = one_vs_all(X, y, num_labels, lambda, num_iterations)
    % Some useful variables
    m = size(X, 1);
    n = size(X, 2);

    % We need to return the following variables correctly
    all_theta = zeros(num_labels, n + 1);

    % Add ones to the X data matrix.
    X = [ones(m, 1) X];

    for class_index=1:num_labels
        % Convert scalar y to vector with related bit being set to 1.
        y_vector = (y == class_index);

        % Set options for fminunc
        options = optimset('GradObj', 'on', 'MaxIter', num_iterations);

        % Set initial thetas to zeros.
        initial_theta = zeros(n + 1, 1);

        % Train the model for current class.
        gradient_function = @(t) gradient_callback(X, y_vector, t, lambda);

        [theta] = fmincg(gradient_function, initial_theta, options);

        % Add theta for current class to the list of thetas.
        theta = theta';
        all_theta(class_index, :) = theta;
    end
end


% Predict the label for a trained one-vs-all classifier. The labels
% are in the range 1..K, where K = size(all_theta, 1)
function p = one_vs_all_predict(all_theta, X)
    m = size(X, 1);
    num_labels = size(all_theta, 1);

    % We need to return the following variables correctly.
    p = zeros(size(X, 1), 1);

    % Add ones to the X data matrix
    X = [ones(m, 1) X];

    % Calculate probabilities of each number for each input example.
    % Each row relates to the input image and each column is a probability that this example is 1 or 2 or 3 etc.
    h = sigmoid(X * all_theta');

    % Now let's find the highest predicted probability for each row.
    % Also find out the row index with highest probability since the index is the number we're trying to predict.
    [p_vals, p] = max(h, [], 2);
end


% [ml-class] Changes Made:
% 1) Function name and argument specifications
% 2) Output display
%

% Read options
if exist('options', 'var') && ~isempty(options) && isfield(options, 'MaxIter')
    length = options.MaxIter;
else
    length = 100;
end


RHO = 0.01;                            % a bunch of constants for line searches
SIG = 0.5;       % RHO and SIG are the constants in the Wolfe-Powell conditions
INT = 0.1;    % don't reevaluate within 0.1 of the limit of the current bracket
EXT = 3.0;                    % extrapolate maximum 3 times the current bracket
MAX = 20;                         % max 20 function evaluations per line search
RATIO = 100;                                      % maximum allowed slope ratio

argstr = ['feval(f, X'];                      % compose string used to call function
for i = 1:(nargin - 3)
  argstr = [argstr, ',P', int2str(i)];
end
argstr = [argstr, ')'];

if max(size(length)) == 2, red=length(2); length=length(1); else red=1; end
S=['Iteration '];

i = 0;                                            % zero the run length counter
ls_failed = 0;                             % no previous line search has failed
fX = [];
[f1 df1] = eval(argstr);                      % get function value and gradient
i = i + (length<0);                                            % count epochs?!
s = -df1;                                        % search direction is steepest
d1 = -s'*s;                                                 % this is the slope
z1 = red/(1-d1);                                  % initial step is red/(|s|+1)

while i < abs(length)                                      % while not finished
  i = i + (length>0);                                      % count iterations?!

  X0 = X; f0 = f1; df0 = df1;                   % make a copy of current values
  X = X + z1*s;                                             % begin line search
  [f2 df2] = eval(argstr);
  i = i + (length<0);                                          % count epochs?!
  d2 = df2'*s;
  f3 = f1; d3 = d1; z3 = -z1;             % initialize point 3 equal to point 1
  if length>0, M = MAX; else M = min(MAX, -length-i); end
  success = 0; limit = -1;                     % initialize quanteties
  while 1
    while ((f2 > f1+z1*RHO*d1) || (d2 > -SIG*d1)) && (M > 0)
      limit = z1;                                         % tighten the bracket
      if f2 > f1
        z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3);                 % quadratic fit
      else
        A = 6*(f2-f3)/z3+3*(d2+d3);                                 % cubic fit
        B = 3*(f3-f2)-z3*(d3+2*d2);
        z2 = (sqrt(B*B-A*d2*z3*z3)-B)/A;       % numerical error possible - ok!
      end
      if isnan(z2) || isinf(z2)
        z2 = z3/2;                  % if we had a numerical problem then bisect
      end
      z2 = max(min(z2, INT*z3),(1-INT)*z3);  % don't accept too close to limits
      z1 = z1 + z2;                                           % update the step
      X = X + z2*s;
      [f2 df2] = eval(argstr);
      M = M - 1; i = i + (length<0);                           % count epochs?!
      d2 = df2'*s;
      z3 = z3-z2;                    % z3 is now relative to the location of z2
    end
    if f2 > f1+z1*RHO*d1 || d2 > -SIG*d1
      break;                                                % this is a failure
    elseif d2 > SIG*d1
      success = 1; break;                                             % success
    elseif M == 0
      break;                                                          % failure
    end
    A = 6*(f2-f3)/z3+3*(d2+d3);                      % make cubic extrapolation
    B = 3*(f3-f2)-z3*(d3+2*d2);
    z2 = -d2*z3*z3/(B+sqrt(B*B-A*d2*z3*z3));        % num. error possible - ok!
    if ~isreal(z2) || isnan(z2) || isinf(z2) || z2 < 0 % num prob or wrong sign?
      if limit < -0.5                               % if we have no upper limit
        z2 = z1 * (EXT-1);                 % the extrapolate the maximum amount
      else
        z2 = (limit-z1)/2;                                   % otherwise bisect
      end
    elseif (limit > -0.5) && (z2+z1 > limit)         % extraplation beyond max?
      z2 = (limit-z1)/2;                                               % bisect
    elseif (limit < -0.5) && (z2+z1 > z1*EXT)       % extrapolation beyond limit
      z2 = z1*(EXT-1.0);                           % set to extrapolation limit
    elseif z2 < -z3*INT
      z2 = -z3*INT;
    elseif (limit > -0.5) && (z2 < (limit-z1)*(1.0-INT))  % too close to limit?
      z2 = (limit-z1)*(1.0-INT);
    end
    f3 = f2; d3 = d2; z3 = -z2;                  % set point 3 equal to point 2
    z1 = z1 + z2; X = X + z2*s;                      % update current estimates
    [f2 df2] = eval(argstr);
    M = M - 1; i = i + (length<0);                             % count epochs?!
    d2 = df2'*s;
  end                                                      % end of line search

  if success                                         % if line search succeeded
    f1 = f2; fX = [fX' f1]';
    fprintf('%s %4i | Cost: %4.6e\r', S, i, f1);
    s = (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;      % Polack-Ribiere direction
    tmp = df1; df1 = df2; df2 = tmp;                         % swap derivatives
    d2 = df1'*s;
    if d2 > 0                                      % new slope must be negative
      s = -df1;                              % otherwise use steepest direction
      d2 = -s'*s;
    end
    z1 = z1 * min(RATIO, d1/(d2-realmin));          % slope ratio but max RATIO
    d1 = d2;
    ls_failed = 0;                              % this line search did not fail
  else
    X = X0; f1 = f0; df1 = df0;  % restore point from before failed line search
    if ls_failed || i > abs(length)          % line search failed twice in a row
      break;                             % or we ran out of time, so we give up
    end
    tmp = df1; df1 = df2; df2 = tmp;                         % swap derivatives
    s = -df1;                                                    % try steepest
    d1 = -s'*s;
    z1 = 1/(1-d1);
    ls_failed = 1;                                    % this line search failed
  end
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
end
fprintf('\n');


% Displays 2D data stored in X.
% Each raw of X is one squared image reshaped into vector.
function [h, display_array] = display_data(X)
	% Since every row in X is a squared image reshaped into vector we may calculate its width.
	example_width = round(sqrt(size(X, 2)));

	% Gray Image
	colormap(gray);

	% Compute rows, cols
	[m n] = size(X);
	example_height = (n / example_width);

	% Compute number of items to display
	display_rows = floor(sqrt(m));
	display_cols = ceil(m / display_rows);

	% Between images padding
	pad = 1;

	% Setup blank display
	display_array = -ones(pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad));

	% Copy each example into a patch on the display array
	curr_ex = 1;
	for j = 1:display_rows
		for i = 1:display_cols
			if curr_ex > m,
				break;
			end
			% Copy the patch

			% Get the max value of the patch
			max_val = max(abs(X(curr_ex, :)));

			row_shift = pad + (j - 1) * (example_height + pad) + (1:example_height);
			column_shift = pad + (i - 1) * (example_width + pad) + (1:example_width);

			display_array(row_shift, column_shift) = reshape(X(curr_ex, :), example_height, example_width) / max_val;
			curr_ex = curr_ex + 1;
		end
		if curr_ex > m,
			break;
		end
	end

	% Display Image
	h = imagesc(display_array, [-1 1]);

	% Do not show axis
	axis image off;

	drawnow;
end
