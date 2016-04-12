function W = stack2vector(par,W,varargin)

if nargin == 1
    Wi = rand(par.nords,par.inputDim);
    bi = zeros(par.nords,1);
    for j = 1:par.hidden_layers-1
        Wh{j}.W = rand(par.nords,par.nords);
        Wh{j}.b = zeros(par.nords,1);
    end
    Wo = rand(par.outputDim,par.nords);
    bo = zeros(par.outputDim,1);
else
    Wi = W.Wi;
    Wh = W.Wh;
    Wo = W.Wo;
end

Whv = [];
bhv = [];
for j = 1:par.hidden_layers-1
    Whv = [Whv Wh{j}.W(:)];
    bhv = [bhv Wh{j}.b(:)];
end

W = [Wi(:);Whv(:);Wo(:);bi(:);bhv(:);bo(:)];
end