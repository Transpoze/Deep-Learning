%Assignment3BatchNorm, DD2424, Addi Djikic, addi@kth.se
clear all;
close all;
clc;
rng(400);
disp('-----Running With Batch-Normalization')


%----------- Load all data for a single batch test
    [X,Y,y] = getTraining('data_batch_1.mat'); 
    [Xvalid,Yvalid,yvalid] = getTraining('data_batch_2.mat');
    [X_Testdata, Y_Testdata, y_Testdata] = getTraining('test_batch.mat');
    mean_X = mean(X, 2);
    X = X - repmat(mean_X, [1, size(X,2)]);
    Xvalid = Xvalid - repmat(mean_X, [1, size(X,2)]);
    X_Testdata = X_Testdata - repmat(mean_X, [1, size(X,2)]);
%-------------    

% %----------- Load all data for all the batches
%     [Xt1,Yt1,yt1] = getTraining('data_batch_1.mat');
%     [Xt2,Yt2,yt2] = getTraining('data_batch_2.mat');
%     [Xt3,Yt3,yt3] = getTraining('data_batch_3.mat');
%     [Xt4,Yt4,yt4] = getTraining('data_batch_4.mat');
%     [Xt5,Yt5,yt5] = getTraining('data_batch_5.mat');
%     [Xtest,Ytest,ytest] = getTraining('test_batch.mat');
%     X_TRAIN = [Xt1,Xt2,Xt3,Xt4,Xt5(:,1:9000)];
%     Y_TRAIN = [Yt1,Yt2,Yt3,Yt4,Yt5(:,1:9000)];
%     y_TRAIN = [yt1;yt2;yt3;yt4;yt5(1:9000)];
% 
%     mean_X_all = mean(X_TRAIN, 2);
%     X_TRAIN = X_TRAIN - repmat(mean_X_all, [1, size(X_TRAIN,2)]);
% 
%     X_VALID = Xt5(:,9001:10000);
%     X_VALID = X_VALID - mean_X_all;
%     Y_VALID = Yt5(:,9001:10000);
%     y_VALID = yt5(9001:10000);
%     Xtest = Xtest - mean_X_all;
%     
%     X = X_TRAIN;
%     Y = Y_TRAIN;
%     y = y_TRAIN;
%     
%     Xvalid = X_VALID;
%     Yvalid = Y_VALID;
%     yvalid = y_VALID;
% %-------------  

% ------------ Set the parameters
    stdDev = 0.001;
    
    lambda = 4.0165e-4;
    %lambda = 0.000001;
    eta = 0.236166;
    
%     lowEta = 0.001;
%     mediumEta = 0.1;
%     highEta = 0.6;
%         eta = highEta;
    
    n_epochs = 20;
    n_batch = 100;
     
    rho = 0.9;
    decay = 0.7; %0.95 default
    
% ------------------------


% ------ Choose amount of layers

M = [50];
LayerSize = size(M,2) + 1

% ------

    
%%--- Uncommet to run function to compare the gradient errors
%      eta = 0.01;
%      lambda = 0;
%      [W_cell, b_cell] = getWeightAndBias(X,Y,stdDev,M);
%      comp = CompareTheGredientsError(X,Y,W_cell,b_cell,0,M);
     
%%--- Perform the Whole MiniBatch Gradient Decent with K-Layer
       [W_cell, b_cell] = getWeightAndBias(X,Y,stdDev,M);
       Xcomp = Xvalid;
       Ycomp = Yvalid;
       ycomp = yvalid;
       
       [Wstar, bstar,meanMoving,varMoving] = MiniBatchGD(X, Y, Xcomp, Ycomp, n_batch, eta, n_epochs, W_cell, b_cell, lambda,rho,decay,M);
       acc = ComputeAccuracy(Xcomp, ycomp, Wstar, bstar, M,meanMoving,varMoving);
       accPercent = acc*100;
       
       disp(['Accuracy is: ', num2str(accPercent), '%'])
         disp('-------Script with batch-norm completed')

% %-----------Set search parameters to find optimal values
%     e_min = log10(0.0750);
%     e_max = log10(0.3810);
%     
%     l_min = log10(6e-5); 
%     l_max = log10(0.0015);  
%     
%     n_epochs = 10;
%     n_batch = 100;
%      
%     rho = 0.9
%     decay = 0.7
%     nbrOfRuns = 60;
% %-----------   
% %%%%%%% Runs the function to find optimal lambda and eta
%         Xbatch = X;
%         Ybatch = Y;
%         XvalidBatch = Xvalid;
%         YvalidBatch = Yvalid;
%         storeMatrixBN=zeros(nbrOfRuns,3);
%         storeMatrixBN = findParameterSpan(n_epochs,n_batch,rho,decay,nbrOfRuns, Xbatch,Ybatch,Xvalid ,Yvalid,yvalid, stdDev,M,e_min,e_max,l_min,l_max);
%         disp('Lambda, Eta and Accuracy - matrix')
%         lambdaEtaAccMatrix = storeMatrixBN
%         save('storeMatrixBN.mat','lambdaEtaAccMatrix');

 %%%%%%%%%%%%%%%%%%%%%% Finding the optimal span for Eta and Lambda
 function storeMatrixBN = findParameterSpan(n_epochs,n_batch,rho,decay,nbrOfRuns, Xbatch,Ybatch,Xvalid ,Yvalid,yvalid, stdDev,M,e_min,e_max,l_min,l_max)

    tic
    for runs = 1:nbrOfRuns
        %%---Initilize the Weight and Bias matricies
            [W_Cell, b_Cell] = getWeightAndBias(Xbatch,Ybatch,stdDev,M);

        e_set = e_min + (e_max - e_min)*rand(1,1);
        eta = 10^e_set;

        l_set = l_min + (l_max - l_min)*rand(1,1);
        lambda = 10^l_set;

        disp(' ');
        disp(['---------------Do Mini Batch Gradient Decent Number: ', num2str(runs), ' Of ', num2str(nbrOfRuns)])
        disp(' ');

        [Wstar, bstar,mean_AV,var_AV] = MiniBatchGD(Xbatch, Ybatch, Xvalid, Yvalid, n_batch, eta, n_epochs, W_Cell, b_Cell, lambda,rho,decay,M);


        finalAccuracy(runs) = ComputeAccuracy(Xvalid, yvalid, Wstar, bstar,M,mean_AV,var_AV);
        currentAccuracy = finalAccuracy(runs);
        currentAccuracyPercent = currentAccuracy*100;

        disp(' ');
        disp(['Accuracy for current run: ', num2str(currentAccuracyPercent), '%'])
        disp(' ');

        storeMatrixBN(runs,1) = lambda;
        storeMatrixBN(runs,2) = eta;
        storeMatrixBN(runs,3) = currentAccuracy;

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
function [W_cell, b_cell] = getWeightAndBias(X,Y,stdDev,M)
    
    K = size(Y,1);
    d = size(X,1);
    
    %Allocate the cells
    W_cell = cell(1,size(M,2) + 1);
    b_cell = cell(1,size(M,2) + 1);
    
    %Set the first cells
    W_cell{1} = randn(M(1),d).*stdDev;
    b_cell{1} = zeros(M(1),1);
    
    %Sett all nodes inbetween
    for i = 2:size(M,2)   
       W_cell{i} = randn(M(i),M(i-1)).*stdDev;
       b_cell{i} = zeros(M(i),1);         
    end

    %Set the last layer node
    W_cell{size(M,2)+1} = randn(K,M(end)).*stdDev;
    b_cell{size(M,2)+1} = zeros(K,1);

end

%%%%%%%%%%%%%%%%%%%%%% Returns the softmaxed values from the cells 
function [h, s, sHat, meanScores, varScores, P] = EvaluateClassifier(X, W, b, M,varargin)
 
    s = cell(1,size(M,2)+1);
    h = cell(1,size(M,2)+1);
    n = size(X,2);
    
    %Check if the mean and variance are pre-computed, else set them here
    if isempty(varargin)
        meanScores = cell(1,size(M,2)+1);
        varScores = cell(1,size(M,2)+1);
    end
    if ~isempty(varargin)
        meanScores = varargin{1};
        varScores = varargin{2};
    end
    
    s{1} = W{1}*X + b{1};
    meanScores{1} = (1/n)*sum(s{1},2);
    %meanScores{1} = mean(s{1},2);
    varScores{1} = var(s{1}, 0, 2)*((n-1)/n);
    
    sHat{1} = BatchNormalize(s{1},meanScores{1},varScores{1});
    h{1} = sHat{1}.*(sHat{1}>0);
    
    for i = 2:size(M,2) + 1
        s{i} = W{i}*h{i-1} + b{i};
        
        if isempty(varargin)
            meanScores{i} = (1/n)*sum(s{i},2);
            %meanScores{i} = mean(s{i},2);
            varScores{i} = var(s{i}, 0, 2)*((n-1)/n);
        end
        
        sHat{i} = BatchNormalize(s{i},meanScores{i},varScores{i});
        h{i} = sHat{i}.*(sHat{i}>0);
   
    end
    s{end} = W{end}*h{end-1} + b{end};
    P = softmax(s{end});
end

%%%%%%%%%%%%%%%%%%%%%% Computes the cost value
function J = ComputeCost(X, Y, W, b, lambda,M,varargin)
    nbrOfImages = size(X,2);
    
    %Check if mean and variance are pre-computed
    if isempty(varargin)
        [~,~,~,~,~,p] = EvaluateClassifier(X,W,b,M);
    end
    
    if ~isempty(varargin)
        meanMoving = varargin{1};
        varMoving = varargin{2};
        [~,~,~,~,~,p] = EvaluateClassifier(X,W,b,M,meanMoving,varMoving);
    end
    
    for i = 1:nbrOfImages
        lCross(i) = -log(Y(:,i)'*p(:,i)); 
    end

    for j = 1:size(M,2)+1
        Wij(j) = sum(sum(W{j}.^2));
    end
    
    Wij = sum(Wij);
    J = ((sum(lCross))/nbrOfImages) + lambda*Wij;
end

%%%%%%%%%%%%%%%%%%%%% Computes the mini-batch Gredient
function [grad_W, grad_b,meanScores,varScores] = ComputeGradients(Xmini, Ymini, W, b, lambda,M)
    %slide 31 lecture 4
    
    [h,s,sHat,meanScores,varScores,p] = EvaluateClassifier(Xmini,W,b,M);
    
    dJdb = cell(1,size(M,2) + 1);
    dJdW = cell(1,size(M,2) + 1);
    grad_W = cell(1,size(M,2) + 1);
    grad_b = cell(1,size(M,2) + 1);
    entries = size(Xmini,2);
    
    for k = 1:size(M,2)+1
        dJdb{k} = zeros(size(b{k}));
        dJdW{k} = zeros(size(W{k}));
    end
    
    for q = 1:size(Xmini,2)
        g(:,q) = (-Ymini(:,q)'/(Ymini(:,q)'*p(:,q)))*(diag(p(:,q))-(p(:,q)*p(:,q)'));
    end
    
    %Set layer size and perform backwardpass
    K = size(M,2) + 1;
    dJdW{K} = (g*h{K-1}')/entries;
    dJdb{K} = (sum(g,2))/entries;
    g = W{K}'*g;
    g = g.*(sHat{K-1}>0);
    
    for j = K-1:-1:1
        g = BatchNormBackPass(g,s{j},meanScores{j},varScores{j},entries);
        dJdb{j} = (sum(g,2))/entries;

        if j == 1
            dJdW{j} = (g*Xmini')/entries;
            %No propagation
        end

        if j > 1
            dJdW{j} = (g*h{j-1}')/entries;
            %Propagate
            g = W{j}'*g;
            g = g.*(sHat{j-1}>0);
        end
        
    end
    
    %add regularization term
    for l = 1:size(M,2)+1
        grad_W{l} = (dJdW{l}) + 2*lambda*W{l};
        grad_b{l} = (dJdb{l});
    end

end

%%%%%%%%%%%%%%%%%%%%% Compare the gredients
function comp = CompareTheGredientsError(X,Y,W_cell,b_cell,lambda,M)
    
    %Reduced number of images used for comparing the gradients
    X_train = X(:,1:3);
    Y_train = Y(:,1:3);
    
    disp('Computing my gradients...')
    %My gredient test function
    [grad_W, grad_b,~,~] = ComputeGradients(X_train, Y_train, W_cell, b_cell, lambda, M);
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
    %%Check error in grads slow
        for i = 1:size(M,2)+1
            gradWDiff_slow(i) = max(max(abs(grad_W{i} - ngrad_W_slow{i})));
            gradWDiffSlow = gradWDiff_slow(i)
            gradbDiff_slow(i) = max(max(abs(grad_b{i} - ngrad_b_slow{i})));
            gradbDiffSlow = gradbDiff_slow(i)
        end
    disp('----------------------------------')

    disp('----------------Errors in Num Fast')
    %%Check error in grads fast
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
function [Wstar, bstar,mean_AV,var_AV] = MiniBatchGD(X, Y, Xcomp, Ycomp, n_batch, eta, n_epochs, W, b, lambda,rho,decay,M)
                          
   N = size(X,2);
   Mlays = size(M,2);
   alpha = 0.99;
   firstMeanAndVar = 0;
   [mW, mb] = initMomentum(X,Y,M);
   
   for i = 1:n_epochs
       
       for j = 1:N/n_batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            inds = j_start:j_end;
             
            Xbatch = X(:, inds);
            Ybatch = Y(:, inds);
            
           if firstMeanAndVar == 1
            [grad_W, grad_b, meanScores,varScores] = ComputeGradients(Xbatch, Ybatch, W, b, lambda,M);
            var = varScores;
            mean = meanScores;
           end
           
           if firstMeanAndVar == 0
               [grad_W, grad_b, meanScores,varScores] = ComputeGradients(Xbatch, Ybatch, W, b, lambda,M);
               mean_AV = meanScores;
               mean = mean_AV;
               var_AV = varScores;
               var = var_AV;
               firstMeanAndVar = 1;
           end
            
            %step with momentum
            for p = 1:Mlays + 1
                mW{p} = rho*mW{p} + eta*grad_W{p};
                W{p} = W{p} - mW{p};
                
                mb{p} = rho*mb{p} + eta*grad_b{p};
                b{p} = b{p} - mb{p};
                
                mean_AV{p} = alpha*mean_AV{p} + (1-alpha)*mean{p};
                var_AV{p} = alpha*var_AV{p} + (1-alpha)*var{p};
                
            end
       end
       
       eta = eta*decay;
       
       currCostTrain(i) = ComputeCost(X, Y, W, b, lambda,M,mean_AV,var_AV);
       currLoss = currCostTrain(i);
       currCostValidation(i) = ComputeCost(Xcomp, Ycomp, W, b, lambda,M,mean_AV,var_AV);
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
function acc = ComputeAccuracy(X, y, W, b, M,varargin)
    if isempty(varargin)
        [~,~,~,~,~,p] = EvaluateClassifier(X, W, b,M);
    end
    if ~isempty(varargin)
        [~,~,~,~,~,p] = EvaluateClassifier(X, W, b,M,varargin{1},varargin{2});
    end
    
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

%%%%%%%%%%%%%%%%%%%%% Initilizes the momentum for gradient step
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

%%%%%%%%%%%%%%%%%%%%% Computes the batch normalize for estimated s
function sHat = BatchNormalize(s,mu,var)
    epsilon = 0.001;
    sHat = ((diag(var + epsilon))^(-0.5))*(s-mu); 
end

%%%%%%%%%%%%%%%%%%%%% Get the g (out) that will be used for the batch norm
function [BNBP] = BatchNormBackPass(g,s,muScore,varScore,n)
    epsilon = 0.001;
    Vb =(varScore + epsilon);
    dJdVb = (-0.5)*sum(g.*(Vb).^(-3/2).*(s-muScore),2);
    dJdmub = -(sum(g.*(Vb).^(-0.5),2));
    %g_out is dJds_i = BNBP
    BNBP = g.*(Vb).^(-0.5) + (2/n)*(dJdVb).*(s-muScore) + dJdmub*(1/n); 
end

%%%%%%%%%%%%%%%%%%%%% Central difference method to compute gradients
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h,M)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda,M);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda,M);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda,M);
    
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda,M);
    
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end

end

%%%%%%%%%%%%%%%%%%%%% Finate difference method to compute gradients
function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h,M)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

%[c, ~] = ComputeCost(X, Y, W, b, lambda);
c = ComputeCost(X, Y, W, b, lambda,M);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        %[c2, ~] = ComputeCost(X, Y, W, b_try, lambda);
        c2 = ComputeCost(X, Y, W, b_try, lambda,M);
        grad_b{j}(i) = (c2-c) / h;
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})   
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
       % [c2, ~] = ComputeCost(X, Y, W_try, b, lambda);
        c2 = ComputeCost(X, Y, W_try, b, lambda,M);
        grad_W{j}(i) = (c2-c) / h;
    end
end

end