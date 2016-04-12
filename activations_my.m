function y = activations_my(x,func_name)

    func_name = lower(func_name);
    
    switch func_name
        case 'sigmoid'
            y = sigmoid(x);
        case 'tanh'
            y = tanh(x);
        case 'relu'
            y = max(0,x);
        case 'binary'
            y = ones(size(x));
        otherwise
            disp('Please input:(sigmoid,Relu,Binary)');
    end

end