%Assignment 4, Addi Djikic, addi@kth.se
clear all;
close all;
clc;
%rng(400)
%format long

disp('---------Running RNN....')

%%Read the data
    fileName = 'goblet_book.txt';
    [book_data] = readBookData(fileName);
    
%%HashCode the data and numbers
    [char_to_ind, ind_to_char, K] = hashCodeData(book_data);

%-----------------------SET HYPERPARAMETERS
%-----------------------
m = 100;
seq_length = 25;%Input sequence length
sig = 0.01;
epochs = 8;
eta = 0.1; %Learning rate
epsilon = 1e-10;
syntLength = 200; %Nmbr of characters to predict
prntTextEvery = 500; %Print the synthesized text after these number of itr

%Bias vectors
RNN.b = zeros(m,1);
RNN.c = zeros(K,1);
%Weight matricies
RNN.U = randn(m, K)*sig;
RNN.W = randn(m, m)*sig;
RNN.V = randn(K, m)*sig;
    
%Random character for Dummy vector to test the RNN functions   
    x_init = 'S';
    x0 = getHot(x_init,K,char_to_ind);
    x0 = double(x0);

%Init hidden vector
h0 = zeros(m,1);

%Nmbr of characters we want
n = seq_length; 
%-------------------------------------
%-------------------------------------


%%%%%%--------------Test the functions and check Gradients before SGT 
%%Forward pass
%     e = 1;
%     X_chars = book_data(e:e+seq_length-1);
%     Y_chars = book_data(e+1:e+seq_length);
%     for t = 1:seq_length
%         X(:,t) = getHot(X_chars(:,t),K,char_to_ind);
%         Y(:,t) = getHot(Y_chars(:,t),K,char_to_ind);
%     end
%     [o, h, h0, a, p, X,Y] = forwardPass(seq_length,K,RNN,m, X, Y,h0);
% %%Backward pass
%     grads = BackwardPropagation(seq_length,Y,p,X,h0,h,K,RNN,m,a);
% %%Compare the gradients numerically
%     num_grads = ComputeGradsNum(X, Y, RNN, 1e-4,seq_length,K,m);
%     compGrads = CompareGradients(grads,num_grads,RNN,eta)
%%%%%%-----------------------------------------------------------
%%Test the Synthesize with dummy
    %[SYNT] = synthesize(h0, x0, n, RNN,ind_to_char,char_to_ind,K,m,seq_length);
%%Test the Predict text with dummy
    %%predictedText = getPredictedText(SYNT,seq_length,ind_to_char)
%%%%%%-----------------------------------------------------------    
  

%%%Train the RNN to output the predicted text
    adaGrad = AdaGrad(book_data, seq_length,K,char_to_ind,ind_to_char,RNN,m,epochs,eta,epsilon,syntLength,prntTextEvery)
    

%%The gradient decent function
function adaGrad = AdaGrad(book_data, seq_length,K,char_to_ind,ind_to_char,RNN,m,epochs,eta,epsilon,syntLength,prntTextEvery)
    
    %Reset all step variables
    step = 1;
    smooth_loss = 0;
    smoothStep = 1;
    count = 0;
    syntCount = 0; 
    
    %Allocate the variables
    mm.b = zeros(size(RNN.b));
    mm.c = zeros(size(RNN.c));
    mm.U = zeros(size(RNN.U));
    mm.W = zeros(size(RNN.W));
    mm.V = zeros(size(RNN.V));
    X = zeros(K,seq_length);
    Y = zeros(K,seq_length);
    
    for epo = 1:epochs
        e = 1;
        hprev = zeros(m,1);
        
        for j = 1:size(book_data,2)/seq_length
            
            X_chars = book_data(e:e+seq_length-1);
            Y_chars = book_data(e+1:e+seq_length);
            for t = 1:seq_length
                X(:,t) = getHot(X_chars(:,t),K,char_to_ind);
                Y(:,t) = getHot(Y_chars(:,t),K,char_to_ind);
            end
            
            [~, h, ~, a, p, ~,~] = forwardPass(seq_length,K,RNN,m, X, Y,hprev);
            grads = BackwardPropagation(seq_length,Y,p,X,hprev,h,K,RNN,m,a);
            
            mm.b = mm.b + grads.b.^2;
            mm.c = mm.c + grads.c.^2;
            mm.U = mm.U + grads.U.^2;
            mm.W = mm.W + grads.W.^2;
            mm.V = mm.V + grads.V.^2;
            
            learnRateUpdateb = (eta./sqrt(mm.b + epsilon)).*grads.b;
            learnRateUpdatec = (eta./sqrt(mm.c + epsilon)).*grads.c;
            learnRateUpdateU = (eta./sqrt(mm.U + epsilon)).*grads.U;
            learnRateUpdateW = (eta./sqrt(mm.W + epsilon)).*grads.W;
            learnRateUpdateV = (eta./sqrt(mm.V + epsilon)).*grads.V;
            
            RNN.b = RNN.b - learnRateUpdateb;
            RNN.c = RNN.c - learnRateUpdatec;
            RNN.U = RNN.U - learnRateUpdateU;
            RNN.W = RNN.W - learnRateUpdateW;
            RNN.V = RNN.V - learnRateUpdateV;
            
            [~, ~,~ , ~, p, ~ ,~] = forwardPass(seq_length,K,RNN,m, X, Y,hprev);
            
            
            currLoss = -sum(diag(log(Y'*p)));
            
            if epo == 1 && j == 1
                disp(' ')
                disp('Smooth loss was initialized')
                smooth_loss = currLoss;
            end
            smooth_loss = 0.999*smooth_loss + 0.001*currLoss;
            
            smoothVector(smoothStep) = smooth_loss; %store the smoothloss
            
            step = step + 1;
            if step == 100
                disp(['Current smooth loss: ', num2str(smooth_loss)]);
                step = 1;
            end
            
            
        if syntCount == prntTextEvery
            hotText = synthesize(hprev,X(:,1),RNN,ind_to_char,char_to_ind,K,m,syntLength);
            predictedText = getPredictedText(hotText,syntLength,ind_to_char);
            disp(['Text at step: ',num2str(count)]);
            disp(predictedText);
            syntCount = 0;
        end
            %Update allt the step parameters along with hidden state
            
            hprev = h(:,seq_length);
            
            smoothStep = smoothStep + 1;
            
            syntCount = syntCount + 1;
            
            count = count +1; 
            
            e = e + seq_length;

        end
        
        currentEpoch = epo;
        disp('-------------------------------')
        disp('-------------------------------')
        disp('-------------------------------')
        disp(' ')
        disp(' ')
        disp(['NUMBER OF EPOCHS MADE : ', num2str(currentEpoch)]);
        disp(' ')
        disp(' ')
        disp('-------------------------------')
        disp('-------------------------------')
        disp('-------------------------------')
    end
    
    plot(smoothVector)
    
    adaGrad = 0; %dummy

end

%%Reds all data from the file as a textfile
function [book_data] = readBookData(fileName)
    book_fname = fileName;
    fid = fopen(book_fname,'r');
    book_data = fscanf(fid,'%c'); %All charachters stored
    fclose(fid);
end

%%Encodes the data from the textfile to numbers correspoding to letter (vice versa) 
function [char_to_ind, ind_to_char, K] = hashCodeData(book_data)

    book_chars = unique(book_data)';

    dimInOut = size(book_chars,1);
    K = dimInOut;

    char_to_ind = containers.Map('KeyType','char','ValueType','int32');
    ind_to_char = containers.Map('KeyType','int32','ValueType','char');

    for i = 1:K
        char_to_ind(book_chars(i)) = i;
        ind_to_char(i) = book_chars(i);
    end
end

%%Converts a letter to onehot encoded number
function HotChar = getHot(theChar,K,char_to_ind)
    HotChar = zeros(K,1);
    numOfChar = char_to_ind(theChar);
    HotChar(numOfChar) = 1;
end

%%Decodes a onehot encoded number to a regular number
function ColdCharNum = getCold(oneHotChar)
    ColdCharNum = find(oneHotChar == 1);
end

%%Prints the predicted text
function predictedText = getPredictedText(Y,seq_length,ind_to_char)
    for r = 1:seq_length
        coldNumbers(r) = getCold(Y(:,r));
        coldChar(r) = ind_to_char(coldNumbers(r));
    end
   predictedText = coldChar;
end

%%Synthesizes the text
function [Y] = synthesize(h0, x0, RNN,ind_to_char,char_to_ind,K,m,seq_length)
                         
    x = zeros(K,seq_length);
    Y = zeros(K,seq_length);
    a = zeros(m,seq_length);
    h = zeros(m,seq_length);
    o = zeros(K,seq_length);
    p = zeros(K,seq_length);
    
    for t = 1:seq_length
        if t == 1
            a(:,t) = RNN.W*h0 + RNN.U*x0 + RNN.b;
            h(:,t) = tanh(a(:,t));
            o(:,t) = RNN.V*h(:,t) + RNN.c;
            p(:,t) = softmax(o(:,t));

            cp = cumsum(p(:,t));
            aa = rand;
            ixs = find(cp-aa >0); 
            ii = ixs(1);
            nextX = ind_to_char(ii);
            x(:,t) = getHot(nextX,K,char_to_ind);
            Y(:,t) = x(:,t);
        end

        if t > 1
            a(:,t) = RNN.W*h(:,t-1) + RNN.U*x(:,t-1) + RNN.b;
            h(:,t) = tanh(a(:,t));
            o(:,t) = RNN.V*h(:,t) + RNN.c;
            p(:,t) = softmax(o(:,t));

            cp = cumsum(p(:,t));
            aa = rand;
            ixs = find(cp-aa >0);
            ii = ixs(1);
            nextX = ind_to_char(ii);
            x(:,t) = getHot(nextX,K,char_to_ind);
            Y(:,t) = x(:,t);
        end
    end
end

%%Forward-propagation
function [o, h, h0, a, p, X,Y] = forwardPass(seq_length,K,RNN,m,X,Y,h0)
                                                           
    a = zeros(m,seq_length);
    h = zeros(m,seq_length);
    o = zeros(K,seq_length);
    p = zeros(K,seq_length); 
    
    
   for t = 1:seq_length 
     
        if t == 1
            a(:,t) = RNN.W*h0 + RNN.U*X(:,t) + RNN.b;
            h(:,t) = tanh(a(:,t));
            o(:,t) = RNN.V*h(:,t) + RNN.c;
            p(:,t) = softmax(o(:,t));
        end
        
        if t > 1
            a(:,t) = RNN.W*h(:,t-1) + RNN.U*X(:,t) + RNN.b;
            h(:,t) = tanh(a(:,t));
            o(:,t) = RNN.V*h(:,t) + RNN.c;
            p(:,t) = softmax(o(:,t));
        end
   end
        
end

function currLoss = getLoss(X,Y, RNN,h0,seq_length,K,m)
    [~, ~, ~, ~, p, ~,~] = forwardPass(seq_length,K,RNN,m,X,Y,h0);
    currLoss = -sum(diag(log(Y'*p)));
end

%%Backward-propagation
function [grads] = BackwardPropagation(seq_length,Y,p,X,h0,h,K,RNN,m,a)
    %g = -(Y-p);
    g = p-Y;
    
    dLdo = g;
    dLdh = zeros(m,seq_length);
    grads.V = (g*h');
     
    dLdh(:,seq_length) = RNN.V'*g(:,seq_length);
    dLda(:,seq_length) = dLdh(:,seq_length).*(1 - tanh(a(:,seq_length)).^2); 
    
    for i = seq_length-1:-1:1
        dLdh(:,i) = RNN.V'*dLdo(:,i) + RNN.W'*dLda(:,i+1);
        dLda(:,i) = dLdh(:,i).*(1 - tanh(a(:,i)).^2);
    end
    
    g_a = dLda;
    
    grad_W_h0 = g_a(:,1)*h0';
    grads.W = grad_W_h0 + g_a(:,2:seq_length)*h(:,1:seq_length-1)';
    
    grads.U = g_a*X';
 
    grads.b = sum(g_a,2);
    
    grads.c = sum(g,2);
    
    for f = fieldnames(grads)'
        grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
    end
        
end

%%Compare the gradient errors from the numerical analysis
function gradError = CompareGradients(grads,num_grads,RNN,eta)

    for f = fieldnames(RNN)'
      RNN.(f{1}) = RNN.(f{1}) - eta * grads.(f{1});
    end
    
    gradError.b = max(max(abs(grads.b-num_grads.b)./(max(abs(grads.b),abs(num_grads.b)))));
    gradError.c = max(max(abs(grads.c-num_grads.c)./(max(abs(grads.c),abs(num_grads.c)))));
    gradError.U = max(max(abs(grads.U-num_grads.U)./(max(abs(grads.U),abs(num_grads.U)))));
    gradError.W = max(max(abs(grads.W-num_grads.W)./(max(abs(grads.W),abs(num_grads.W)))));
    gradError.V = max(max(abs(grads.V-num_grads.V)./(max(abs(grads.V),abs(num_grads.V)))));
end
%%Compute gradients numerically
function num_grads = ComputeGradsNum(X, Y, RNN, h, seq_length,K,m)

    for f = fieldnames(RNN)'
        disp('Computing numerical gradient for')
        disp(['Field name: ' f{1} ]);
        num_grads.(f{1}) = ComputeGradNumSlow(X, Y, f{1}, RNN, h, seq_length,K,m);
    end
end
function grad = ComputeGradNumSlow(X, Y, f, RNN, h, seq_length,K,m)

    n = numel(RNN.(f));
    grad = zeros(size(RNN.(f)));
    hprev = zeros(size(RNN.W, 1), 1);
    for i=1:n
        RNN_try = RNN;
        RNN_try.(f)(i) = RNN.(f)(i) - h;
   
        l1 = getLoss(X,Y, RNN_try,hprev,seq_length,K,m);
        RNN_try.(f)(i) = RNN.(f)(i) + h;
        
        l2 = getLoss(X,Y, RNN_try,hprev,seq_length,K,m);
        grad(i) = (l2-l1)/(2*h);
    end
end





