%%%%%%%%%% Assignment1, Addi Djikic, TSCRM1, addi@kth.se 
clear all;
close all;
clc;

rng(400)
%format long;

%Load all data
[XX,YY,yy] = getTraining('data_batch_1.mat');
[Xvalid,Yvalid,yvalid] = getTraining('data_batch_2.mat');
[X_Testdata, Y_TestData, y_Testdata] = getTraining('test_batch.mat');


%%-------- Set all parameters
    K = 10;
    d = 3072;
    N = 10000;
    
    stdDev = 0.01;
    W = randn(K,d).*stdDev;
    b = randn(K,1).*stdDev;
    
    lambda = 1;
    n_epochs = 40;
    n_batch = 100;
    eta = 0.01; 
%%--------


disp('------------------------------------')
    J_Cost_Test = ComputeCost(XX,YY,W,b,lambda)
disp('------------------------------------')

disp('------------------------------------')
    networkAccuracy = ComputeAccuracy(XX,yy,W,b)
disp('------------------------------------')
disp('------------------------------------')

%Uncomment this section below to test the gradient errors

% %%%%%%%%%%%%%%%%%%%%%% Compute gredients, and check errors less than 1e-6
% 
%     X_train = XX(:,1);
%     Y_train = YY(:,1);
% 
%     %My gredient test function
%         P_in = EvaluateClassifier(X_train, W, b);
%         [grad_W, grad_b] = ComputeGradients(X_train,Y_train,P_in,W,lambda);
% 
%     %%NumSlow
%         [ngrad_b, ngrad_W] = ComputeGradsNumSlow(X_train,Y_train,W,b,lambda,1e-6);
%     %%NumFast
%         [ngrad_b_fast, ngrad_W_fast] = ComputeGradsNum(X_train, Y_train, W, b, lambda, 1e-6);
% 
%     disp(' ')
%     disp(' ')
% 
%     disp('----------------Errors in Num Slow')
%     %%Check error in grads slow
%         errorGrad_W_slow = max(max(abs(grad_W - ngrad_W)))
%         errorGrad_b_slow = max(max(abs(grad_b - ngrad_b)))
%         disp('------------------------------------')
% 
%     disp('----------------Errors in Num Fast')
%     %%Check error in grads fast
%         errorGrad_W_fast = max(max(abs(grad_W - ngrad_W_fast)))
%         errorGrad_b_fast = max(max(abs(grad_b - ngrad_b_fast)))
%         disp('------------------------------------')
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


disp(' ------------- Compute mini-batch gradient decent ---------------')

[Wstar, bstar] = MiniBatchGD(XX, YY, Xvalid, Yvalid, n_batch, eta, n_epochs, W, b, lambda);

disp('------------------------------------')
    finalAccuracy = ComputeAccuracy(Xvalid, yvalid, Wstar, bstar)
disp('------------------------------------')

%---Show the reshaped Wstar images
    montage(getShapedWstarimages(Wstar))
    

%%%%%%%%%%%%%%%%%%%%%% Re-arrange the W matrix for display
function WmatrixArranged = getShapedWstarimages(Wstar)
    for i=1:10
        im = reshape(Wstar(i, :), 32, 32, 3);
        s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
        s_im{i} = permute(s_im{i}, [2, 1, 3]);
    end 
    WmatrixArranged = s_im;
end


%%%%%%%%%%%%%%%%%%%%%% Function to read the image data 
function [X Y y] = getTraining(trainData)
    addpath Datasets/cifar-10-batches-mat/;
    A = load(trainData);

    X = double(A.data')./255;

    Y_pre = double(A.labels' + 1);

    for i = 1:10000
        Y(:,i) = Y_pre(i).*ones(1,10) == 1:10;
    end

    y = A.labels+1; 
end



%%%%%%%%%%%%%%%%%%%%%% Returns the softmaxed values
function P = EvaluateClassifier(XX, W, b)
    s = W*XX + b;
    P = exp(s)./(sum(exp(s))); %or -> P = softmax(s);
end
    


%%%%%%%%%%%%%%%%%%%%%% Computes the cost value
function J = ComputeCost(XX, YY, W, b, lambda)
    
    nbrOfImages = size(XX,2);
    p = EvaluateClassifier(XX,W,b);

    for i = 1:nbrOfImages %size(XX,2)
        J_pre(i) = -log(YY(:,i)'*p(:,i)); 
    end

    J = sum(J_pre)/nbrOfImages + lambda*sum(sum(W.^2));
end




%%%%%%%%%%%%%%%%%%%%%% Computes the network accuracy
function acc = ComputeAccuracy(X, y, W, b)

    [~, argmax] = max(EvaluateClassifier(X, W, b));
    k_star = argmax;
    nmrOfCorrect = 0;

    for i = 1:size(X,2)
        if (k_star(1,i) == y(i,:))
            nmrOfCorrect = nmrOfCorrect + 1;
        end
    end

    acc = nmrOfCorrect/size(X,2);

end


%%%%%%%%%%%%%%%%%%%%%%% Mini batch gradient decent 
function [Wstar, bstar] = MiniBatchGD(X, Y, Xcomp, Ycomp, n_batch, eta, n_epochs, W, b, lambda)

   N = size(X,2);
   
    for i = 1:n_epochs
        for j=1:N/n_batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            %inds = j_start:j_end;
            
            Xbatch = X(:, j_start:j_end);
            Ybatch = Y(:, j_start:j_end);
            
            P_ev = EvaluateClassifier(Xbatch,W,b);
            
            [grad_W, grad_b] = ComputeGradients(Xbatch,Ybatch,P_ev,W,lambda);
            W = W - eta*grad_W;
            b = b - eta*grad_b;
        end
        
        currentCost(i) = ComputeCost(X, Y, W, b, lambda);
        CurrentCostLoss = currentCost(i)
       
        currentCostValidation(i) = ComputeCost(Xcomp, Ycomp, W, b, lambda);
        
    end
    
    disp('------------------------------------')
        firstCost = currentCost(1)
    disp('------------------------------------')
    
    %---Plot the datas
        
        plot(currentCost)
        hold on
        plot(currentCostValidation)
        legend('Training Loss','Validation Loss')
        xlabel('Epochs')
        ylabel('Loss')
        hold off
    
    Wstar = W;
    bstar = b;
    
end


%%%%%%%%%%%%%%%%%%%%%% Computes the mini-batch gredient 
function [grad_W, grad_b] = ComputeGradients(XminiB, YminiB, P, W, lambda)

    %From slide 91 & 93 in Lecure 3 DD2424
    
    partialDeriv_b = 0;
    partialDeriv_W = 0;
    
    for i = 1:size(XminiB,2)
        grad_W_pre = -(YminiB(:,i)'/(YminiB(:,i)'*P(:,i)))*(diag(P(:,i)) - P(:,i)*P(:,i)');
        partialDeriv_W = partialDeriv_W + grad_W_pre'*XminiB(:,i)';   
    end
    
   
    grad_W = (1/size(XminiB,2)).*partialDeriv_W + (2*lambda).*W;

    for j = 1:size(XminiB,2)
        grad_b_pre = -(YminiB(:,i)'/(YminiB(:,i)'*P(:,i)))*(diag(P(:,i)) - P(:,i)*P(:,i)');
        partialDeriv_b = partialDeriv_b + grad_b_pre';
    end
    
    grad_b = (1/size(XminiB,2))*(partialDeriv_b);
    
end






