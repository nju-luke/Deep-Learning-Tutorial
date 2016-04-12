clear;clc;

%% Load DataSet

addpath ../MINIST;
addpath(genpath('../minFunc'));
addpath ../functions;

load = [];
load.num = 2000;        % Annotate this line to load all the samples;
load.type = '1D';       % choose the sample type '1D','2D'

[images,labels,test_images,test_labels,imageDim] = LoadData(load);

classes = length(unique(labels));

%%  Initialize the parameters
par.inputDim = imageDim;
par.hidden_layers = 2;
par.nords = 100;
par.outputDim = classes;
par.act_fun = 'sigmoid';           % choose one of (sigmoid,tanh,Relu,Binary)
par.lambda = 1e-4;

options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';
options.MaxIter=100;
%
theta_ini = stack2vector(par);                  %Initialize the Weights as a structure var

%% check Numerical Gradient
theta = theta_ini;
if true
% if false
    [~, grad] = MLP_cost(theta,images,labels,par);
    numgrad = computeNumericalGradient( @(x)MLP_cost(x,images,labels,par),theta);

    diff = norm(numgrad-grad)/norm(numgrad+grad);
    disp(diff);
end

%%  calculate the gradient by minFunc
theta = theta_ini;
opt_theta = minFunc(@(W)MLP_cost(W,images,labels,par),theta,options);
% opt_theta = fminunc(@(W)MLP_cost(W,images,labels,par),theta,options);

[~,~,preds] = MLP_cost(opt_theta,test_images,test_labels,par,'test');
accu = sum(preds == test_labels)/size(test_labels,1);
disp('Calculate the gradient by minFunc:');
disp(accu);

%%  calculate the gradient by SGD
% theta = theta_ini;
% opt_theta = SGD(theta,images,labels,par);
% [~,~,preds] = MLP_cost(opt_theta,test_images,test_labels,par,'test');
% accu = sum(preds == test_labels)/size(test_labels,1);
% disp('calculate the gradient by SGD:');
% disp(accu);