

clc;clear;close all


addpath('functions')
addpath('functions/logistic')
addpath('dataset')


%% cancer data pre-processing

X = csvread(  'wdbc.data');
Y = X(:,2);
X = X(:,[3:end]);  % column 1 is index, column 2 is lable


m  = repmat(mean(X,1),length(Y),1);
X = X>=m;



[M,~] = size(X);

rng(50)
train_idx = datasample([1:M],floor(0.75*M),'Replace',false);
test_idx = setdiff([1:M],train_idx);

X_train = X(train_idx,:);
Y_train = Y(train_idx,1);
X_test = X(test_idx,:);
Y_test = Y(test_idx,1);


%% non-private baseline, i.e., logistic regression on the non-private data

[~, P] = size(X_train);
initial_theta = rand(P + 1, 1);


X_train_extend = [  X_train  ones(  size(X_train,1)  ,1  )];

[theta, cost] = fminunc(@(t)(objFunction(t, X_train_extend, Y_train)), initial_theta);
X_test_extend = [   X_test   ones(  size(X_test,1)  ,1  )];
Y_pred =  sigmoid( X_test_extend  * theta )>0.5 ;
acc_nonprivate= regression_accuracy(Y_pred,Y_test)


%% set privacy budget
epsilon = 0.1000 ; %   0.2000    0.3000    0.4000    0.5000    0.6000    0.7000    0.8000    0.9000    1.0000
C = 30; %   C trials


%% using XOR
fD = X_train ;
[N,P] = size( fD );

sf = P;

alpha = 1; % or 0.75

T = 3.5*pi;

ACC = [];
ERR = [];

block_num = 6;

size1  = N/block_num; % divide the query dataset into 10 blocks, each with size 24 x 16
size2 = P;

for c = 1:C % repeat C trials and take the average
    
    
    B = zeros(N,P);
    
    for i = 1:block_num % divide into blocks
        tic; B_block =   EHMC_MVB_sampler(size1,size2,sf,epsilon/block_num,alpha, T);toc;
        B(  [ 1+(i-1)*size1 : i*size1 ] , : ) = B_block;
    end
    
    fD_xored = xor(B,fD);
    
    err = (   norm(fD_xored-fD,'fro')/norm( double(fD),'fro')  )^2;
    ERR  =[ERR err];
    
    X_train_xored = fD_xored;
    initial_theta = rand(P +1 , 1);
    
    X_train_xored_extend = [  X_train_xored  ones(  size(X_train_xored,1)  ,1  )];
    [theta, cost] = ...
        fminunc(@(t)(objFunction(t, X_train_xored_extend, Y_train )), initial_theta);
    
    X_test_extend = [   X_test   ones(  size(X_test,1)  ,1  )];
    Y_pred =  sigmoid( X_test_extend  * theta )>0.5 ;
    acc = regression_accuracy(Y_pred,Y_test);
    ACC = [ACC acc];
end

err_xor = mean(ERR)
acc_xor = mean(ACC)


%% using privacy-preserving logistic regression via objective perturbation



ACC_PPLR = [];
for j = 1:C
    initial_theta = rand(P+1 , 1);
    
    
    theta = PPLR( [X_train  ones(size(X_train,1),1) ],Y_train,epsilon,initial_theta);
    
    Y_pred =  sigmoid(  [X_test ones(size(X_test,1),1) ] * theta )>0.5 ;
    acc = regression_accuracy(Y_pred,Y_test);
    ACC_PPLR = [ACC_PPLR acc];
end


acc_PPLR = mean(ACC_PPLR)
