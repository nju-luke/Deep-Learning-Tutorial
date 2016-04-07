% Load MNIST DataSet
% [images,labels,test_images,test_labels,imageDim] = LoadData(бо2D' ); %Load for 2D
% [images,labels,test_images,test_labels,imageDim] = LoadData( '1D' ); %Load for 1D

function [images,labels,test_images,test_labels,imageDim] = LoadData(type,num)
%% Load Data

images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

images = images(:,1:num);
labels = labels(1:num);

test_images = loadMNISTImages('t10k-images-idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
test_labels(labels==0) = 10; % Remap 0 to 10
imageDim = size(images,1);

test_images = test_images(:,1:num);
test_labels = test_labels(1:num);

if type == '2D'
    imageDim = sqrt(imageDim);
    images = reshape(images,imageDim,imageDim,[]);
    test_images = reshape(test_images,imageDim,imageDim,[]);
end
