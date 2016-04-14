function  [Wi,bi,ci,Wo,bo] = RBM_fit(theta,images,labels,par,varargin)
%%
num = size(images,2);
[Wi,Whb,Wo,bi,bo] = vector2mat(theta,par);
%%
ci = zeros(par.inputDim,1);
delta_Wi = 0;
delta_bi = 0;
delta_ci = 0;
delta_Wo = 0;
delta_bo = 0;

%%
%Divide the samples to num_subset
index_r = randperm(num,num);
dim_subset = num/par.num_subset;

labels_mat = full(sparse(labels,1:size(labels,1),1));

for j = 1:par.num_subset
    index_j = index_r((j-1)*dim_subset+1:j*dim_subset);
    image_set{j} = images(:,index_j);
    labels_set{j} = labels_mat(:,index_j);
end

for epoch = 1:par.maxepoch
    
    for ind_sub = 1:par.num_subset
        
        % Gibs k times sampling
        visual_layer0 = image_set{ind_sub};
        visual_layerk = visual_layer0;
        
        labels_0 = labels_set{ind_sub};
        labels_k = labels_0;
        
        sig_h = sigmoid(bsxfun(@plus,Wi*visual_layerk+Wo'*labels_k,bi));     % The probility for the first sampling of hidden layer                           
        hidd_layerk = double(rand(size(sig_h))<sig_h);
        sig_h0 = sig_h;
        for k = 1:par.CD_k-1
            sig_v = sigmoid(bsxfun(@plus,Wi'*hidd_layerk,ci));
            visual_layerk = double(rand(size(sig_v))<sig_v);

            labels_k = softmax_my(bsxfun(@plus,Wo*hidd_layerk,bo));
            [~,I] = max(labels_k);
            labels_k = zeros(size(labels_k));            
            labels_k(sub2ind(size(labels_k),I,1:size(labels_k,2))) = 1;
            
            sig_h = sigmoid(bsxfun(@plus,Wi*visual_layerk+Wo'*labels_k,bi));
            hidd_layerk = double(rand(size(sig_h))<sig_h);
        end
        
        delta_Wi = par.momentum*delta_Wi + (sig_h0*visual_layer0' - sig_h*visual_layerk')/dim_subset - par.lambda*Wi;
        delta_bi = par.momentum*delta_bi + (sum(sig_h0,2) - sum(sig_h,2))/dim_subset - par.lambda*bi;
        delta_ci = par.momentum*delta_ci + (sum(visual_layer0,2) - sum(visual_layerk,2))/dim_subset - par.lambda*ci;
        delta_Wo = par.momentum*delta_Wo + (labels_0*sig_h0' - labels_k*sig_h')/dim_subset -par.lambda*Wo;
        delta_bo = par.momentum*delta_bo + (sum(labels_0,2) - sum(labels_k,2))/dim_subset - par.lambda*bo;
        
        Wi = Wi + par.alpha*delta_Wi;
        bi = bi + par.alpha*delta_bi;
        ci = ci + par.alpha*delta_ci;
        Wo = Wo + par.alpha*delta_Wo;
        bo = bo + par.alpha*delta_bo;
%         
    end
if mod(epoch,25) == 0
    display_network(visual_layerk);
    title(['epoch = ',num2str(epoch)]);
end

end

end

