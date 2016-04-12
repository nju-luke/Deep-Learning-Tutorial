function [J, grad] = softmax_cost(W,X,imageDim,labels_mat,lambda)

W = reshape(W, imageDim, []);
probs = softmax_my(W'*X);

J = -labels_mat.*log(probs)+lambda*sum(W(:).^2)/2;
J = sum(J(:));
grad_W = X*(labels_mat-probs)' + lambda*W;
grad = grad_W(:);
end

