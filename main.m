clear;clc;

%% Load DataSet
addpath ../MINIST
addpath ../minFunc/

num = 1000;
[images,labels,test_images,test_labels,imageDim] = LoadData('1D',num);
classes = length(unique(labels));

lambda = 1e-6;

theta = rand(imageDim,classes);

% Here we use two different methods to learn the weight
%%
% 1. Use the minfunc to fit
addpath ../minFunc;
labels_mat = full(sparse(labels,1:num,ones(1,num)));
options.Display = 'iter';
options.MaxIterations = 10;
theta_1 = theta(:);
% theta = fminunc(@(W)softmax(W,images,imageDim,labels_mat,lambda),theta,options);
theta = minFunc(@(W)softmax(W,images,imageDim,labels_mat,lambda),theta_1,options);
theta = reshape(theta,imageDim,[]);
%%
% 2. The stochastic gradient decent

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
 
%% Show the accuracy
prob = exp(theta'*images);
prob = bsxfun(@rdivide,prob,sum(prob));
[~,prob_label] = max(prob);
disp('The accuracy from the train set:')
accu = 1-sum(prob_label==labels')/num;
disp(accu)

prob = exp(theta'*test_images);
prob = bsxfun(@rdivide,prob,sum(prob));
[~,prob_label] = max(prob);
disp('The accuracy from the test set:')
accu = 1-sum(prob_label==test_labels')/num;
disp(accu)

