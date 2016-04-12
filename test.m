prob = softmax_my(theta'*images);
[~,prob_label] = max(prob);
disp('The accuracy from the train set:')
accu = 1-sum(prob_label==labels')/num;
disp(accu)

prob = softmax_my(theta'*test_images);
[~,prob_label] = max(prob);
disp('The accuracy from the test set:')
accu = 1-sum(prob_label==test_labels')/num;
disp(accu)

