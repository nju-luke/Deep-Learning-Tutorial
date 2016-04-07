function [J, grad] = softmax(W,X,labels_mat,lambda)

h_theta = exp(W'*X);
probs = bsxfun(@rdivide,h_theta,sum(h_theta));

J = -labels_mat.*log(probs);
J = sum(J(:));
grad_W = X*(labels_mat-probs)' + lambda*W;
grad = grad_W(:);
end

