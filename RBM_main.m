clear;clc;

%% Load DataSet

addpath ../MINIST;
addpath(genpath('../minFunc'));
addpath ../functions;
%%
load = [];
% load.num = 1000;        % Annotate this line to load all the samples;
load.type = '1D';       % choose the sample type '1D','2D'

[images,labels,test_images,test_labels,imageDim] = LoadData(load);

%%

% images = sampleIMAGES;imageDim = size(images,1);
% images = images(:,1:2000);
% labels = images;
classes = length(unique(labels));

%%  Initialize the parameters
par = [];
par.inputDim = imageDim;
par.hidden_layers = 1;
par.nords = 100;
par.outputDim = classes;
par.act_fun = 'sigmoid';           % choose one of (sigmoid,tanh,Relu,Binary)
par.lambda = 1e-2;
par.maxepoch = 1000;
par.num_subset = 20;
par.CD_k = 4;
par.alpha = 1e-2;
par.momentum = 0.2;


%%  
theta = stack2vector(par);                  %Initialize the Weights as a structure var

[Wi,bi,ci,Wo,bo] = RBM_fit(theta,images,labels,par);
%%
figure
display_network(Wi');

H = sigmoid(bsxfun(@plus,Wi*images,bi));
V = sigmoid(bsxfun(@plus,Wi'*H,ci));
V = V(:,1:100);
figure
display_network(V);

%%
labels_prob = softmax_my(bsxfun(@plus,Wo*H,bo));
[~,labels_vect] = max(labels_prob);

accu = sum(labels_vect'==labels)/max(size(labels));
fprintf('The accuracy of the traing data is %f\n',accu);

%%
Ht = sigmoid(bsxfun(@plus,Wi*test_images,bi));
labels_prob = softmax_my(bsxfun(@plus,Wo*Ht,bo));
[~,labels_vect_t] = max(labels_prob);

accu = sum(labels_vect_t'==test_labels)/max(size(test_labels));
fprintf('The accuracy of the traing data is %f\n',accu);



