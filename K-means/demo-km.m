% Initialization
clear; close all; clc;

% Clustering Set #1 

% Loading the data.
fprintf('Loading the data set #1...\n');
load('set1.mat');

% Plotting the data.
fprintf('Plotting the data set #1...\n');
subplot(2, 2, 1);
plot(X(:, 1), X(:, 2), 'k+','LineWidth', 1, 'MarkerSize', 7);
title('Training Set #1');
