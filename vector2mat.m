function [Wi,Whb,Wo,bi,bo] = vector2mat(W,par)

Whb = {};

index_end = par.inputDim*par.nords;
Wi = reshape(W(1:par.inputDim*par.nords),par.nords,[]);

for j = 1:par.hidden_layers-1
    index_start = index_end+1;
    index_end = index_end+(par.nords)^2;
    Whb{j}.W = reshape(W(index_start:index_end),par.nords,[]);
end

index_start = index_end+1;
index_end = index_end+par.nords*par.outputDim;
Wo = reshape(W(index_start:index_end),par.outputDim,[]);


index_start = index_end+1;
index_end = index_end+par.nords;
bi = W(index_start:index_end);

for j = 1:par.hidden_layers-1
    index_start = index_end+1;
    index_end = index_end+par.nords;
    Whb{j}.b = W(index_start:index_end);
end

bo = W(index_end+1:end);
end