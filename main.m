clear;clc;

%% Load DataSet

addpath ../MINIST;
addpath(genpath('../minFunc'));
addpath ../functions;
%%
load = [];
load.num = 1000;        % Annotate this line to load all the samples;
load.type = '1D';       % choose the sample type '1D','2D'

[images,~,~,~,imageDim] = LoadData(load);
%%

% images = sampleIMAGES;imageDim = size(images,1);
% images = images(:,1:2000);
labels = images;
classes = size(images,1);

%%  Initialize the parameters
par = [];
par.inputDim = imageDim;
par.hidden_layers = 1;
par.nords = 100;
par.outputDim = classes;
par.act_fun = 'Sigmoid';           % choose one of (sigmoid,tanh,Relu,Binary)
%

%%  calculate the gradient by minFunc
theta = stack2vector(par);                  %Initialize the Weights as a structure var

%% check Numerical Gradient

% if true
if false
    [~, grad] = AE_cost(theta,images,labels,par);
    numgrad = computeNumericalGradient( @(x)AE_cost(x,images,labels,par),theta);

    diff = norm(numgrad-grad)/norm(numgrad+grad);
    disp(diff);
end

%%
par.lambda = 1e-4;
par.rho = 0.05;
par.beta = 3;

options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';
options.MaxIter=1000;

%
[opt_theta,cost] = minFunc(@(W)AE_cost(W,images,labels,par),theta,options);
%
[Wi,Whb,Wo,bi,bo] = vector2mat(opt_theta,par);
figure
display_network(Wi');