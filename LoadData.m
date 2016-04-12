% Load MNIST DataSet
% [images,labels,test_images,test_labels,imageDim] = LoadData(para); 
% para.type = '1D' or '2D'
% para.num is the numbers of samples

function [images,labels,test_images,test_labels,imageDim] = LoadData(para)
%% Load Data

images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');
labels(labels == 0) = 10;

test_images = loadMNISTImages('t10k-images-idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
test_labels(test_labels == 0) = 10;
imageDim = size(images,1);

if isfield(para,'num')
    num = para.num;
    images = images(:,1:num);
    labels = labels(1:num);
    test_images = test_images(:,1:num);
    test_labels = test_labels(1:num);
end

if strcmp(para.type, '2D')
    imageDim = sqrt(imageDim);
    images = reshape(images,imageDim,imageDim,[]);
    test_images = reshape(test_images,imageDim,imageDim,[]);
end

