function y = activations_d(x,func_name)

    func_name = lower(func_name);
    
    switch func_name
        case 'sigmoid'
            y = sigmoid(x).*(1-sigmoid(x));
        case 'tanh'
            y = 1 - tanh(x).^2;
        case 'relu'
             y = double(x>0);
        case 'binary'
            y = 1;
        otherwise
            disp('Please input:(sigmoid,Relu,Binary)');
    end

end