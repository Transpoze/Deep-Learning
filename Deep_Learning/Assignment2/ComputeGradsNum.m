function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

%[c, ~] = ComputeCost(X, Y, W, b, lambda);
c = ComputeCost(X, Y, W, b, lambda);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        %[c2, ~] = ComputeCost(X, Y, W, b_try, lambda);
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b{j}(i) = (c2-c) / h;
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})   
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
       % [c2, ~] = ComputeCost(X, Y, W_try, b, lambda);
        c2 = ComputeCost(X, Y, W_try, b, lambda);
        grad_W{j}(i) = (c2-c) / h;
    end
end

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