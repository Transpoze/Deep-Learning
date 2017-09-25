%%%%%%%%%% Assignment2, Addi Djikic, TSCRM1, addi@kth.se 
clear all;
close all;
clc;

%Load all data for a single batch test
    [X,Y,y] = getTraining('data_batch_1.mat'); 
    [Xvalid,Yvalid,yvalid] = getTraining('data_batch_2.mat');
    [X_Testdata, Y_Testdata, y_Testdata] = getTraining('test_batch.mat');
    mean_X = mean(X, 2);
    X = X - repmat(mean_X, [1, size(X,2)]);
    Xvalid = Xvalid - mean_X;
    X_Testdata = X_Testdata - mean_X;

%%------- Load and store all the batch data
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
%%-------

% ------------ Set all parameters
    stdDev = 0.001;
    m = 50;
    %lambda = 0.000001;
    n_epochs = 200;
    n_batch = 100;
    %eta = 0.01; 
    rho = 0.9;
    decay = 0.95;
    
    e_min = log10(0.0220);
    e_max = log10(0.0235);
    l_min = log10(9.731e-4); 
    l_max = log10(9.739e-4); 
    
    optEta = 0.022743553254431392;
    optLambda = 9.731885172855308e-4;
    
    nbrOfRuns = 50;
% ------------------------

 
 %%%%%%% Perform the minibatch gradient decent with optimal values
     Xtrain = X_TRAIN;%X;
     Ytrain = Y_TRAIN;%Y;
     Xcompare = X_VALID;%X_Testdata;
     Ycompare = Y_VALID;%Y_Testdata;
     ycompare = y_VALID;%y_Testdata;

     finalAcc = runMiniBatchGDWithOptimalParameters(n_epochs,n_batch,rho,decay,Xtrain,Ytrain,Xcompare,Ycompare,ycompare,stdDev,m,optEta,optLambda)

%%%%%%% Runs the function to find optimal lambda and eta
    %     Xbatch = X;
    %     Ybatch = Y;
    %     XvalidBatch = Xvalid;
    %     YvalidBatch = Yvalid;
    %     storeMatrix=zeros(nbrOfRuns,3);
    %     storeMatrix = runEpochsMinibatchGDFindSpan(n_epochs,n_batch,rho,decay,nbrOfRuns, Xbatch,Ybatch,Xvalid ,Yvalid,yvalid, stdDev,m,e_min,e_max,l_min,l_max);
    %     disp('Lambda, Eta and Accuracy - matrix')
    %     lambdaEtaAccMatrix = storeMatrix
    %     save('storeMatrix.mat','lambdaEtaAccMatrix');


%%--- Test the evaluate classifier and cost function
    %[~,~,~,P_test] = EvaluateClassifier(X, W_Cell, b_Cell);
    %JCost = ComputeCost(X,Y,W_Cell,b_Cell,lambda);
    
%%--- Uncommet to compare the gradient errors
%     [W_cell, b_cell] = getWeightAndBias(X,Y,stdDev,m);
%     comp = CompareTheGredientsError(X,Y,W_cell,b_cell,0);

function finalAccuracy_withOptimal = runMiniBatchGDWithOptimalParameters(n_epochs,n_batch,rho,decay,X,Y,Xcompare,Ycompare,ycompare,stdDev,m,optEta,optLambda)
     %%---Initilize the Weight and Bias matricies
        [W_Cell, b_Cell] = getWeightAndBias(X,Y,stdDev,m);

    disp(' ');
    disp('---------------Do Mini Batch Gradient Decent With Optimal Values')
    disp(' ');

    [Wstar, bstar] = MiniBatchGD(X, Y, Xcompare, Ycompare, n_batch, optEta, n_epochs, W_Cell, b_Cell, optLambda,rho,decay);

    finalAccuracy_withOptimal = ComputeAccuracy(Xcompare, ycompare, Wstar, bstar)


end
    
    
function storeMatrix = runEpochsMinibatchGDFindSpan(n_epochs,n_batch,rho,decay,nbrOfRuns, Xbatch,Ybatch,Xvalid ,Yvalid,yvalid, stdDev,m,e_min,e_max,l_min,l_max)

    tic
    for runs = 1:nbrOfRuns
        %%---Initilize the Weight and Bias matricies
            [W_Cell, b_Cell] = getWeightAndBias(Xbatch,Ybatch,stdDev,m);

        e_set = e_min + (e_max - e_min)*rand(1,1);
        eta = 10^e_set;

        l_set = l_min + (l_max - l_min)*rand(1,1);
        lambda = 10^l_set;

        disp(' ');
        disp(['---------------Do Mini Batch Gradient Decent Number: ', num2str(runs), ' Of ', num2str(nbrOfRuns)])
        disp(' ');

        [Wstar, bstar] = MiniBatchGD(Xbatch, Ybatch, Xvalid, Yvalid, n_batch, eta, n_epochs, W_Cell, b_Cell, lambda,rho,decay);


        finalAccuracy(runs) = ComputeAccuracy(Xvalid, yvalid, Wstar, bstar);
        currentAccuracy = finalAccuracy(runs);
        currentAccuracyPercent = currentAccuracy*100;

        disp(' ');
        disp(['Accuracy for current run: ', num2str(currentAccuracyPercent), '%'])
        disp(' ');

        storeMatrix(runs,1) = lambda;
        storeMatrix(runs,2) = eta;
        storeMatrix(runs,3) = currentAccuracy;

    end
    toc
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


%%%%%%%%%%%%%%%%%%%%%% Set the weights and bias randomly and return cell
function [W_cell, b_cell] = getWeightAndBias(X,Y,stdDev,m)
    

    K = size(Y,1);
    d = size(X,1);
   
    W1 = randn(m,d).*stdDev;
    b1 = zeros(m,1);
    
    W2 = randn(K,m).*stdDev;
    b2 = zeros(K,1);
     
    W_cell = {W1,W2};
    b_cell = {b1,b2};
    
    
end


%%%%%%%%%%%%%%%%%%%%%% Returns the softmaxed values from the cells
function [s1, h, s, P] = EvaluateClassifier(X, W, b)
    s1 = W{1}*X + b{1};
    h = s1.*(s1>0);
    s = W{2}*h + b{2};
    P = softmax(s);
end


%%%%%%%%%%%%%%%%%%%%%% Computes the cost value
function J = ComputeCost(X, Y, W, b, lambda)
    
    nbrOfImages = size(X,2);
    [~,~,~,p] = EvaluateClassifier(X,W,b);

    for i = 1:nbrOfImages
        lCross(i) = -log(Y(:,i)'*p(:,i)); 
    end

    Wij = sum(sum(W{1}.^2)) + sum(sum(W{2}.^2));
    
    J = ((sum(lCross))/nbrOfImages) + lambda*Wij;
end


%%%%%%%%%%%%%%%%%%%%% Computes the mini-batch gredient, slide 14 lecture 4
function [grad_W, grad_b] = ComputeGradients(Xmini, Ymini, W, b, lambda)
    
    dL_db1 = zeros(size(b{1}));
    dL_dW1 = zeros(size(W{1}));
    dL_db2 = zeros(size(b{2}));
    dL_dW2 = zeros(size(W{2}));
    [s1,h,~,p] = EvaluateClassifier(Xmini,W,b);
    entries = size(Xmini,2);
    
    for i = 1:size(Xmini,2)
       g = (-Ymini(:,i)'/(Ymini(:,i)'*p(:,i)))*(diag(p(:,i))-(p(:,i)*p(:,i)'));
       
       dL_db2 = dL_db2 + g';
       dL_dW2 = dL_dW2 + g'*h(:,i)';
       %Propagate
       g = g*W{2};
       g = g*diag(s1(:,i)>0);
       
       dL_db1 = dL_db1 + g';
       dL_dW1 = dL_dW1 + g'*Xmini(:,i)';
       
    end
    %Devide by entries and return grad as cell
    grad_b1 = dL_db1/entries;
    grad_W1 = dL_dW1/entries + 2*lambda*W{1};
    grad_b2 = dL_db2/entries;
    grad_W2 = dL_dW2/entries + 2*lambda*W{2};
    
    grad_W = {grad_W1, grad_W2};
    grad_b = {grad_b1, grad_b2};
    
end   

%%%%%%%%%%%%%%%%%%%%%%% Mini batch gradient decent 
function [Wstar, bstar] = MiniBatchGD(X, Y, Xcomp, Ycomp, n_batch, eta, n_epochs, W, b, lambda,rho,decay)

   N = size(X,2);
   
   %Initialize momentum
   vt_W1 = zeros(size(W{1}));
   vt_b1 = zeros(size(b{1}));
   vt_W2 = zeros(size(W{2}));
   vt_b2 = zeros(size(b{2}));
   vtW = {vt_W1, vt_W2};
   vtb = {vt_b1, vt_b2};
   
    for i = 1:n_epochs
        
        for j=1:N/n_batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            %inds = j_start:j_end;
            
            Xbatch = X(:, j_start:j_end);
            Ybatch = Y(:, j_start:j_end);
            
            [grad_W, grad_b] = ComputeGradients(Xbatch,Ybatch,W,b,lambda);
            
            vtW{1} = rho*vtW{1} + eta*grad_W{1};
            W{1} = W{1} - vtW{1};
            
            vtb{1} = rho*vtb{1} + eta*grad_b{1};
            b{1} = b{1} - vtb{1};
            
            vtW{2} = rho*vtW{2} + eta*grad_W{2};
            W{2} = W{2} - vtW{2};
            
            vtb{2} = rho*vtb{2} + eta*grad_b{2};
            b{2} = b{2} - vtb{2};        
           
        end
        
        eta = eta*decay;
        
        currentCostTrain(i) = ComputeCost(X, Y, W, b, lambda);
        CurrentCostLoss = currentCostTrain(i);
        currentCostValidation(i) = ComputeCost(Xcomp, Ycomp, W, b, lambda);
        CurrentValidCost = currentCostValidation(i);
        
        disp(['Epoch: ',num2str(i), ' Current Train Cost: ', num2str(CurrentCostLoss), ' Current Validation Cost: ', num2str(CurrentValidCost)]);
        
    end
    
%     disp('------------------------------------')
%         firstCost = currentCostTrain(1)
%     disp('------------------------------------')
    
    %---Plot the datas
        
        plot(currentCostTrain)
        hold on
        plot(currentCostValidation)
        legend('Training Loss','Validation Loss')
        xlabel('Epochs')
        ylabel('Loss')
        hold off     
    
    Wstar = W;
    bstar = b;
    
end

%%%%%%%%%%%%%%%%%%%%% Computes the network accuracy
function acc = ComputeAccuracy(X, y, W, b)

    [~,~,~,p] = EvaluateClassifier(X, W, b);
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

%%%%%%%%%%%%%%%%%%%%% Compare the gredients
function comp = CompareTheGredientsError(X,Y,W_Cell,b_Cell,lambda)

    X_train = X(:,1);
    Y_train = Y(:,1);
    
    disp('Computing own gradients...')
    %My gredient test function
    [grad_W, grad_b] = ComputeGradients(X_train, Y_train, W_Cell, b_Cell, lambda);
    disp('Own gradients computed.')
    disp(' ')
    disp('Computing gradients with slow function...')
    %NumSlow
        [ngrad_b_slow, ngrad_W_slow] = ComputeGradsNumSlow(X_train,Y_train,W_Cell,b_Cell,lambda,1e-5);
    disp('Computing gradients with fast function...')
    %NumFast
        [ngrad_b_fast, ngrad_W_fast] = ComputeGradsNum(X_train, Y_train, W_Cell, b_Cell, lambda, 1e-5);

    disp(' ')
    disp(' ')

    disp('----------------Errors in Num Slow')
    %Check error in grads slow
        errorGrad_W1_slow = max(max(abs(grad_W{1} - ngrad_W_slow{1})))
        errorGrad_b1_slow = max(max(abs(grad_b{1} - ngrad_b_slow{1})))
        errorGrad_W2_slow = max(max(abs(grad_W{2} - ngrad_W_slow{2})))
        errorGrad_b2_slow = max(max(abs(grad_b{2} - ngrad_b_slow{2})))
        disp('------------------------------------')

    disp('----------------Errors in Num Fast')
    %Check error in grads fast
        errorGrad_W1_fast = max(max(abs(grad_W{1} - ngrad_W_fast{1})))
        errorGrad_b1_fast = max(max(abs(grad_b{1} - ngrad_b_fast{1})))
        errorGrad_W2_fast = max(max(abs(grad_W{2} - ngrad_W_fast{2})))
        errorGrad_b2_fast = max(max(abs(grad_b{2} - ngrad_b_fast{2})))
        disp('------------------------------------')
       comp = 0;
end
