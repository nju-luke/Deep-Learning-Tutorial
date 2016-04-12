clear;clc;

%% Load DataSet
addpath ../MINIST;
addpath(genpath('../minFunc'));
addpath ../functions;

load.num = 1000;        % Annotate this line to load all the samples;
load.type = '1D';       % choose the sample type '1D','2D'

[images,labels,test_images,test_labels,imageDim] = LoadData(load);
num = size(labels,1);
classes = length(unique(labels));
% Here we use two different methods to learn the weight
%%
% 1. Use the minfunc to fit
labels_mat = full(sparse(labels,1:num,ones(1,num)));
options.Display = 'iter';
options.MaxIterations = 10;
lambda = 1e-4;
theta_ini = randn(imageDim,classes);
theta_1 = theta_ini(:);
% theta = fminunc(@(W)softmax(W,images,imageDim,labels_mat,lambda),theta,options);
theta = minFunc(@(W)softmax_cost(W,images,imageDim,labels_mat,lambda),theta_1);
theta = reshape(theta,imageDim,[]);
disp('The minfunc fitting:')
test
%%
% 2. The stochastic gradient decent
theta = theta_ini;
delta = inf;
learn_rate = 1e-3;
n = 1;
theta_new = theta;
 while delta > 1e-12 
    index_j = randi(num);
    prob = exp(theta'*images(:,index_j));
    prob = prob/sum(prob);
    
    label = zeros(classes,1);
    label(labels(index_j)) = 1;
    
    grad_theta = images(:,index_j)*(label-prob)'+lambda*theta;
    theta_new = theta - learn_rate*grad_theta;
    delta = sum((theta_new(:)-theta(:)).^2);

%     grad_theta = images(:,index_j)*(1-prob(labels(index_j)));
%     theta_new(:,labels(index_j)) = theta(:,labels(index_j)) - learn_rate*grad_theta;
%     delta = sum((theta_new(:)-theta(:)).^2);

    theta = theta_new;
    n = n+1;
 end
 %
 disp('The SGD fitting:')
 test
