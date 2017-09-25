%%%%% Assignment 3 DD2424 Addi Djikic, addi@kth.se
clear all;
close all;
clc;
format long
rng(400)
disp('-----Running WITHOUT Batch-Normalization')

% %----------- Load all data for a single batch test
%     [X,Y,y] = getTraining('data_batch_1.mat'); 
%     [Xvalid,Yvalid,yvalid] = getTraining('data_batch_2.mat');
%     [X_Testdata, Y_Testdata, y_Testdata] = getTraining('test_batch.mat');
%     mean_X = mean(X, 2);
%     X = X - repmat(mean_X, [1, size(X,2)]);
%     Xvalid = Xvalid - repmat(mean_X, [1, size(X,2)]);
%     X_Testdata = X_Testdata - repmat(mean_X, [1, size(X,2)]);
% %-------------    

%----------- Load all data for all all the batches
    [Xt1,Yt1,yt1] = getTraining('data_batch_1.mat');
    [Xt2,Yt2,yt2] = getTraining('data_batch_2.mat');
    [Xt3,Yt3,yt3] = getTraining('data_batch_3.mat');
    [Xt4,Yt4,yt4] = getTraining('data_batch_4.mat');
    [Xt5,Yt5,yt5] = getTraining('data_batch_5.mat');
    [Xtest,Ytest,ytest] = getTraining('test_batch.mat');
    X_TRAIN = [Xt1,Xt2,Xt3,Xt4,Xt5(:,1:9000)];
    Y_TRAIN = [Yt1,Yt2,Yt3,Yt4,Yt5(:,1:9000)];
    y_TRAIN = [yt1;yt2;yt3;yt4;yt5(1:9000)];

    mean_X_all = mean(X_TRAIN, 2);
    X_TRAIN = X_TRAIN - repmat(mean_X_all, [1, size(X_TRAIN,2)]);

    X_VALID = Xt5(:,9001:10000);
    X_VALID = X_VALID - mean_X_all;
    Y_VALID = Yt5(:,9001:10000);
    y_VALID = yt5(9001:10000);
    Xtest = Xtest - mean_X_all;
    
    X = X_TRAIN;
    Y = Y_TRAIN;
    y = y_TRAIN;
    
    Xvalid = X_VALID;
    Yvalid = Y_VALID;
    yvalid = y_VALID;
%-------------  
    
    
    % ------------ Set all parameters
    stdDev = 0.001;
    lambda = 0.000001;
    %eta = 0.01;
    lowEta = 0.001;
    mediumEta = 0.1;
    highEta = 0.3;
        eta = highEta;
  
    n_epochs = 10;
    n_batch = 100;
     
    rho = 0.9;
    decay = 0.7; %0.95 default
    
    %e_min = log10(0);
    %e_max = log10(0);
    %l_min = log10(0); 
    %l_max = log10(0); 
    %lambda = 0.000001;
    %eta = 0.01; 
    %optEta = 0.022743553254431392;
    %optLambda = 9.731885172855308e-4;
    nbrOfRuns = 50;
% ------------------------

% ------ Choose amount of layers
M = [50];
LayerSize = size(M,2) + 1
% ------


%%--- Test the evaluate classifier and cost function
    %[W_cell, b_cell] = getWeightAndBias(X,Y,stdDev,M);
    %[h,s,P_test] = EvaluateClassifier(X, W_cell, b_cell, M);
    %JCost = ComputeCost(X,Y,W_cell,b_cell,lambda,M)
    
    %[grad_W, grad_b] = ComputeGradients(X(:,1), Y(:,1), W_cell, b_cell, lambda,M)
    
%%--- Uncommet to compare the gradient errors
     %[W_cell, b_cell] = getWeightAndBias(X,Y,stdDev,M);
     %comp = CompareTheGredientsError(X,Y,W_cell,b_cell,0,M);
     
%%--- Perform mini-batch gradient decent
       [W_cell, b_cell] = getWeightAndBias(X,Y,stdDev,M);
       Xcomp = Xvalid;
       Ycomp = Yvalid;
       ycomp = yvalid;
       [Wstar, bstar] = MiniBatchGD(X, Y, Xcomp, Ycomp, n_batch, eta, n_epochs, W_cell, b_cell, lambda,rho,decay,M);
       acc = ComputeAccuracy(Xcomp, ycomp, Wstar, bstar, M);
       accPercent = acc*100;
       disp(['Accuracy is: ', num2str(accPercent), '%'])

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

%%%%%%%%%%%%%%%%%%%%%% Set the weights and bias randomly and return cell
function [W_cell, b_cell] = getWeightAndBias(X,Y,stdDev,M)
    
    K = size(Y,1);
    d = size(X,1);
    
    %Allocate the cells
    W_cell = cell(1,size(M,2) + 1);
    b_cell = cell(1,size(M,2) + 1);
    
    for i = 2:size(M,2)   
       W_cell{i} = randn(M(i),M(i-1)).*stdDev;
       b_cell{i} = zeros(M(i),1);         
    end

    W_cell{1} = randn(M(1),d).*stdDev;
    b_cell{1} = zeros(M(1),1);
    W_cell{size(M,2)+1} = randn(K,M(end)).*stdDev;
    b_cell{size(M,2)+1} = zeros(K,1);
    
end

%%%%%%%%%%%%%%%%%%%%%% Returns the softmaxed values from the cells 
function [h, s, P] = EvaluateClassifier(X, W, b, M)
    
    s = cell(1,size(M,2)+1);
    h = cell(1,size(M,2)+1);

    s{1} = W{1}*X + b{1};
    h{1} = s{1}.*(s{1}>0);
    
    for i = 2:size(M,2) + 1
        %h{i-1} = s{i-1}.*(s{i-1}>0);
        s{i} = W{i}*h{i-1} + b{i};
        h{i} = s{i}.*(s{i}>0);
        %h{i} = max(0,s{i});
    end
    
    P = softmax(s{end});
end

%%%%%%%%%%%%%%%%%%%%%% Computes the cost value
function J = ComputeCost(X, Y, W, b, lambda,M)
    
    nbrOfImages = size(X,2);
    [~,~,p] = EvaluateClassifier(X,W,b,M);

    for i = 1:nbrOfImages
        lCross(i) = -log(Y(:,i)'*p(:,i)); 
    end

    for i = 1:size(M,2)+1
        Wij(i) = sum(sum(W{i}.^2));
    end
    
    Wij = sum(Wij);
    J = ((sum(lCross))/nbrOfImages) + lambda*Wij;
end

%%%%%%%%%%%%%%%%%%%%% Computes the mini-batch Gredient
function [grad_W, grad_b] = ComputeGradients(Xmini, Ymini, W, b, lambda,M)
    %slide 31 lecture 4
    
    [h,s,p] = EvaluateClassifier(Xmini,W,b,M);
    entries = size(Xmini,2);
    
    dJdb = cell(1,size(M,2) + 1);
    dJdW = cell(1,size(M,2) + 1);
    grad_W = cell(1,size(M,2) + 1);
    grad_b = cell(1,size(M,2) + 1);
    
    for k = 1:size(M,2)+1
        dJdb{k} = zeros(size(b{k}));
        dJdW{k} = zeros(size(W{k}));
    end
    
    
    for i = 1:size(Xmini,2)
        g = (-Ymini(:,i)'/(Ymini(:,i)'*p(:,i)))*(diag(p(:,i))-(p(:,i)*p(:,i)'));
        
        for j = size(M,2)+1:-1:1
            dJdb{j} = dJdb{j} + g';
            
            if j == 1
                dJdW{j} = dJdW{j} + g'*Xmini(:,i)'; %+ 2*lambda*W{j};
            end
            
            if j > 1 
                dJdW{j} = dJdW{j} + g'.*h{j-1}(:,i)';
                %Propagate
                g = g*W{j};
                g = g*diag(s{j-1}(:,i)>0);
            end
        end
    end
    
    %add regularization term and devide by number of entries
    for l = 1:size(M,2)+1
        grad_W{l} = (dJdW{l}/entries) + 2*lambda*W{l};
        grad_b{l} = (dJdb{l}/entries);
    end

end

%%%%%%%%%%%%%%%%%%%%% Compare the gredients
function comp = CompareTheGredientsError(X,Y,W_cell,b_cell,lambda,M)

    X_train = X(:,2);
    Y_train = Y(:,2);
    
    disp('Computing my gradients...')
    %My gredient test function
    [grad_W, grad_b] = ComputeGradients(X_train, Y_train, W_cell, b_cell, lambda, M);
    disp('-------- My gradients computed.')
    disp(' ')
    disp('Computing gradients with slow function...')
    %NumSlow
        [ngrad_b_slow, ngrad_W_slow] = ComputeGradsNumSlow(X_train,Y_train,W_cell,b_cell,lambda,1e-5,M);
    disp('---------Num slow done.')
    disp('Computing gradients with faster function...')
    %NumFast
        [ngrad_b_fast, ngrad_W_fast] = ComputeGradsNum(X_train, Y_train, W_cell, b_cell, lambda, 1e-5,M);
    disp('---------Num fast done.')
    disp(' ')
    disp(' ')

    disp('----------------Errors in Num Slow')
    %Check error in grads slow
        for i = 1:size(M,2)+1
            gradWDiff_slow(i) = max(max(abs(grad_W{i} - ngrad_W_slow{i})));
            gradWDiffSlow = gradWDiff_slow(i)
            gradbDiff_slow(i) = max(max(abs(grad_b{i} - ngrad_b_slow{i})));
            gradbDiffSlow = gradbDiff_slow(i)
        end
    disp('----------------------------------')

    disp('----------------Errors in Num Fast')
    %Check error in grads fast
        for j = 1:size(M,2)+1
            gradWDiff_fast(j) = max(max(abs(grad_W{j} - ngrad_W_fast{j})));
            gradWDiffFast = gradWDiff_fast(j)
            gradbDiff_fast(j) = max(max(abs(grad_b{j} - ngrad_b_fast{j})));
            gradbDiffFast = gradbDiff_fast(j)
        end
    disp('----------------------------------')
    
    comp = 0;
end

%%%%%%%%%%%%%%%%%%%%%%% Mini batch gradient decent 
function [Wstar, bstar] = MiniBatchGD(X, Y, Xcomp, Ycomp, n_batch, eta, n_epochs, W, b, lambda,rho,decay,M)
                          
   N = size(X,2);
   Mlays = size(M,2);
   
   %mW = cell(1,Mlays + 1);
   %mb = cell(1,Mlays + 1);
   
   [mW, mb] = initMomentum(X,Y,M);
   
   for i = 1:n_epochs
       
       for j = 1:N/n_batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            inds = j_start:j_end;
             
            Xbatch = X(:, inds);
            Ybatch = Y(:, inds);
            
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, W, b, lambda,M);
           
            %step with momentum
            for p = 1:Mlays + 1
                mW{p} = rho*mW{p} + eta*grad_W{p};
                W{p} = W{p} - mW{p};
                
                mb{p} = rho*mb{p} + eta*grad_b{p};
                b{p} = b{p} - mb{p};
            end
       end
       
       eta = eta*decay;
       
       currCostTrain(i) = ComputeCost(X, Y, W, b, lambda,M);
       currLoss = currCostTrain(i);
       currCostValidation(i) = ComputeCost(Xcomp, Ycomp, W, b, lambda,M);
       currValidLoss = currCostValidation(i);
       
       disp(['Epoch: ',num2str(i), ' Train Cost: ', num2str(currLoss), ' Validation Cost: ', num2str(currValidLoss)]);
   end
       disp('------------------------------------')
        firstCostofTraining = currCostTrain(1)
       disp('------------------------------------')
   
       %---Plot the datas
        
         plot(currCostTrain)
         hold on
         plot(currCostValidation)
         legend('Training Loss','Validation Loss')
         xlabel('Epochs')
         ylabel('Loss')
         hold off     
     
     Wstar = W;
     bstar = b;
 
end

%%%%%%%%%%%%%%%%%%%%% Computes the network accuracy
function acc = ComputeAccuracy(X, y, W, b, M)

    [~,~,p] = EvaluateClassifier(X, W, b,M);
    [~, argmax] = max(p);
    k_star = argmax;
    nmrOfCorrect = 0;

    for i = 1:size(X,2)
        if (k_star(1,i) == y(i,:))
            nmrOfCorrect = nmrOfCorrect + 1;
        end
    end

    acc = nmrOfCorrect/size(X,2);

end

function [mW, mb] = initMomentum(X,Y,M)
    
    K = size(Y,1);
    d = size(X,1);
    
    %Allocate the cells
    mW = cell(1,size(M,2) + 1);
    mb = cell(1,size(M,2) + 1);
    
    for i = 2:size(M,2)   
       mW{i} = zeros(M(i),M(i-1));
       mb{i} = zeros(M(i),1);         
    end

    mW{1} = zeros(M(1),d);
    mb{1} = zeros(M(1),1);
    
    mW{size(M,2)+1} = zeros(K,M(end));
    mb{size(M,2)+1} = zeros(K,1);
end
