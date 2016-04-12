function [J,grad,preds] = MLP_cost(theta,images,labels,par,varargin)
%%

num = size(images,2);

J = 0;
grad = [];
preds = [];


[Wi,Whb,Wo,bi,bo] = vector2mat(theta,par);
%%
zh{1} = bsxfun(@plus,Wi*images,bi);
ah{1} = activations_my(zh{1},par.act_fun);

pow2Wh = 0;
for j = 2:par.hidden_layers
    zh{j} = bsxfun(@plus,Whb{j-1}.W*ah{j-1},Whb{j-1}.b);
    ah{j} = activations_my(zh{j},par.act_fun);
    pow2Wh = pow2Wh + sum(Whb{j-1}.W(:).^2);
end

z_output = bsxfun(@plus,Wo*ah{par.hidden_layers},bo); 
probs = softmax_my(z_output);

% A = probs(isnan(probs));
% probs(isnan(probs)) = 1 - rand(size(A))*0.1;
%
%%
if nargin == 5    
    [~,I] = max(probs);
    preds = I';
    return;
end
%%
labels_mat = full(sparse(labels,1:num,ones(num,1)));
J = -labels_mat.*log(probs);
J = sum(J(:))/num+par.lambda*(sum(Wi(:).^2)+ pow2Wh + sum(Wo(:).^2))/2;
%%

delta_output = -(labels_mat-probs);
delta_h = cell(1,par.hidden_layers);
delta_h{par.hidden_layers} = Wo'*delta_output.*activations_d(zh{par.hidden_layers},par.act_fun);
for j = 1:par.hidden_layers-1
    delta_h{par.hidden_layers-j} = Whb{par.hidden_layers-j}.W'*delta_h{par.hidden_layers-j+1}.*activations_d(zh{par.hidden_layers-j},par.act_fun);
end

%%

grad_Wi = delta_h{1}*images'/num + par.lambda * Wi;
grad_bi = sum(delta_h{1},2)/num;

%%
grad_h.W = [];
grad_h.b = [];
for j = 1:par.hidden_layers-1
    grad_Whb{j}.W = delta_h{j+1}*ah{j}'/num+  par.lambda * Whb{j}.W;
    grad_Whb{j}.b = sum(delta_h{j+1},2)/num;
    grad_h.W = [grad_h.W;grad_Whb{j}.W(:)];
    grad_h.b = [grad_h.b;grad_Whb{j}.b(:)];
end
%%
grad_Wo = delta_output*ah{par.hidden_layers}'/num + par.lambda*Wo;
grad_bo = sum(delta_output,2)/num;
%%
grad = [grad_Wi(:) ;grad_h.W ;grad_Wo(:); grad_bi(:); grad_h.b; grad_bo(:)];
end
