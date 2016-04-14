clear;clc;

%% Load DataSet

addpath ../MINIST;
addpath(genpath('../minFunc'));
addpath ../functions;
%%
load = [];
% load.num = 6000;        % Annotate this line to load all the samples;
load.type = '1D';       % choose the sample type '1D','2D'

[images,~,~,~,imageDim] = LoadData(load);
%%

% images = sampleIMAGES;imageDim = size(images,1);
% images = images(:,1:2000);
labels = images;
mask_ratiao = 0.8;
images = images.*(rand(size(images))<mask_ratiao);
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
par.rho = 0.005;
par.beta = 3;

options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';
options.MaxIter=10000;

%
[theta,cost] = minFunc(@(W)AE_cost(W,images,labels,par),theta,options);
%
[Wi,Whb,Wo,bi,bo] = vector2mat(theta,par);
figure
display_network(Wi');
%%
B = Wi*images;
AO = Wo*B;
%%
index_r = randperm(size(images,2),49);
A_s = images(:,index_r);
display_network_bi(A_s);
AO_s = AO(:,index_r);
figure
display_network_bi(AO_s);
