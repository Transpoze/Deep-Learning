% function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h,M)
% 
% grad_W = cell(numel(W), 1);
% grad_b = cell(numel(b), 1);
% 
% for j=1:length(b)
%     grad_b{j} = zeros(size(b{j}));
%     
%     for i=1:length(b{j})
%         
%         b_try = b;
%         b_try{j}(i) = b_try{j}(i) - h;
%         c1 = ComputeCost(X, Y, W, b_try, lambda,M);
%         
%         b_try = b;
%         b_try{j}(i) = b_try{j}(i) + h;
%         c2 = ComputeCost(X, Y, W, b_try, lambda,M);
%         
%         grad_b{j}(i) = (c2-c1) / (2*h);
%     end
% end
% 
% for j=1:length(W)
%     grad_W{j} = zeros(size(W{j}));
%     
%     for i=1:numel(W{j})
%         
%         W_try = W;
%         W_try{j}(i) = W_try{j}(i) - h;
%         c1 = ComputeCost(X, Y, W_try, b, lambda,M);
%     
%         W_try = W;
%         W_try{j}(i) = W_try{j}(i) + h;
%         c2 = ComputeCost(X, Y, W_try, b, lambda,M);
%     
%         grad_W{j}(i) = (c2-c1) / (2*h);
%     end
% end
% 
% end
% 
% 
% %%%%%%%%%%%%%%%%%%%%%% Computes the cost value
% function J = ComputeCost(X, Y, W, b, lambda,M)
%     
%     nbrOfImages = size(X,2);
%     [~,~,~,~,~,p] = EvaluateClassifier(X,W,b,M);
% 
%     for i = 1:nbrOfImages
%         lCross(i) = -log(Y(:,i)'*p(:,i)); 
%     end
% 
%     for j = 1:size(M,2)+1
%         Wij(j) = sum(sum(W{j}.^2));
%     end
%     
%     Wij = sum(Wij);
%     J = ((sum(lCross))/nbrOfImages) + lambda*Wij;
% end
% 
% %%%%%%%%%%%%%%%%%%%%%% Returns the softmaxed values from the cells 
% function [h, s, sHat, meanScores, varScores, P] = EvaluateClassifier(X, W, b, M)
%     
%     s = cell(1,size(M,2)+1);
%     h = cell(1,size(M,2)+1);
%     n = size(X,2);
%     meanScores = cell(1,size(M,2)+1);
%     varScores = cell(1,size(M,2)+1);
%     
%     s{1} = W{1}*X + b{1};
%     meanScores{1} = (1/n)*sum(s{1},2);
%     %meanScores{1} = mean(s{1},2);
%     varScores{1} = var(s{1}, 0, 2)*((n-1)/n);
%     
%     sHat{1} = BatchNormalize(s{1},meanScores{1},varScores{1});
%     h{1} = sHat{1}.*(sHat{1}>0);
%     
%     for i = 2:size(M,2) + 1
%         
%         s{i} = W{i}*h{i-1} + b{i};
%         meanScores{i} = (1/n)*sum(s{i},2);
%         %meanScores{i} = mean(s{i},2);
%         varScores{i} = var(s{i}, 0, 2)*((n-1)/n);
%         
%         sHat{i} = BatchNormalize(s{i},meanScores{i},varScores{i});
%         h{i} = sHat{i}.*(sHat{i}>0);
%    
%     end
%     s{end} = W{end}*h{end-1} + b{end};
%     P = softmax(s{end});
% end
% 
% 
% 
% % % %%%%%%%%%%%%%%%%%%%%%% Computes the cost value
% % % function J = ComputeCost(X, Y, W, b, lambda,M)
% % %     
% % %     nbrOfImages = size(X,2);
% % %     [~,~,p] = EvaluateClassifier(X,W,b,M);
% % % 
% % %     for i = 1:nbrOfImages
% % %         lCross(i) = -log(Y(:,i)'*p(:,i)); 
% % %     end
% % % 
% % %     for i = 1:size(M,2)+1
% % %         Wij(i) = sum(sum(W{i}.^2));
% % %     end
% % %     
% % %     Wij = sum(Wij);
% % %     J = ((sum(lCross))/nbrOfImages) + lambda*Wij;
% % % end
% % % 
% % % %%%%%%%%%%%%%%%%%%%%%% Returns the softmaxed values from the cells
% % % function [h, s, P] = EvaluateClassifier(X, W, b, M)
% % %     
% % %     s = cell(1,size(M,2)+1);
% % %     h = cell(1,size(M,2)+1);
% % % 
% % %     s{1} = W{1}*X + b{1};
% % %     h{1} = s{1}.*(s{1}>0);
% % %     
% % %     for i = 2:size(M,2) + 1
% % %         %h{i-1} = s{i-1}.*(s{i-1}>0);
% % %         s{i} = W{i}*h{i-1} + b{i};
% % %         h{i} = s{i}.*(s{i}>0);
% % %         
% % %     end
% % %     
% % %     P = softmax(s{end});
% % % end
% 
% 
