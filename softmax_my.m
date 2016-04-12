function probs = softmax_my(X)

h_theta = exp(X);
probs = bsxfun(@rdivide,h_theta,sum(h_theta));

end

