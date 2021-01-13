clear;clc;close all

addpath('dataset/email')
addpath('functions')
addpath('functions/graph_related_functions')

%% email preprocessing
edgeList = csvread('email-Eu-core.csv')+1;
N = max(max(edgeList));
A = full(sparse(edgeList(:, 1), edgeList(:, 2), 1, N, N));
A = A+A';
A = A-diag(diag(A));
A(A>0)=1;

isolated = find(sum(A)==0);
A(isolated,:) = [];
A(:,isolated) = [];

[N,P]  = size(A);

N = size(A,1);
node_label = csvread('email-Eu-core-department-labels.csv')+1;
node_label(isolated,:) = [];
node_label(:,1) = ([1:N])';
K  = 42; % num of communities

G = graph(A);
edgelist = G.Edges;
edgelist = edgelist{:,:};
edgelist = edgelist(:,[1 2]);

comm = zeros(N,K);
ind = sub2ind( [N,K],([1:N])',node_label(:,2));
comm(ind) = 1; % ground truth community affiliation

%% nonprivate network statistics
rng(100)
edge_num_np = sum(A(:))/2;
network_pair_d  = distances(G);
dia_np = max(network_pair_d(:));
density_np = edge_num_np * 2/(N*(N-1));
avg_pl_np = mean(network_pair_d(:));

C_P = bigclam(A, K);

f1 = average_f1(C_P,comm);

%% calculating using XOR on email graph
alpha= 1 ;T = 3.5*pi;sf=2;

epsilon = 0.1000 ; %   0.2000    0.3000    0.4000    0.5000    0.6000    0.7000    0.8000    0.9000    1.0000
kl   = []; avg_f1 = []; edge_num_xor = []; diameter = []; density = []; average_pl = [];
for count = 1:Count
    
    block_n = 29;
    block_p = 29;
    
    size1  = N/block_n;  
    size2 = P/block_p;
    
    B = zeros(N,P);
    for i = 1:block_n
        for j = 1:block_p % divide into blocks
            tic; B_block =   EHMC_MVB_sampler(size1,size2,sf,epsilon/(block_n*block_p),alpha, T);toc;
            B( [ 1+(i-1)*size1 : i*size1 ]  , [ 1+(j-1)*size2 : j*size2 ]  ) = B_block;
        end
    end
    % takes about 20 minutes

    A_xored = xor(A,B);
    A_xored = and(A_xored,A_xored');
    A_xored = A_xored-diag(  diag(A_xored)   );
    G = graph(A_xored);
    % 1) edge numbers
    edge_num =   sum(A_xored(:))/2 ;
    edge_num_xor = [edge_num_xor edge_num];
    
    
    % 2) network diameter
    network_pair_d  = distances(G);
    network_pair_d(find(isinf(network_pair_d)))=0;
    diameter = [diameter max(network_pair_d(:))];
    
    
    % 3) network density
    density = [density edge_num*2/(N*(N-1))];
    
    
    % 4) network average pl
    average_pl = [average_pl mean(network_pair_d(:))];
    
    
    % 5) kl divergence of degree distribution
    kl  = [kl  kldiv(A,A_xored)];

    
    % 6) community detection f1 score
    C_P = bigclam(A_xored, K);
    avg_f1 = [avg_f1 average_f1(C_P,comm)];
    
    count
    
end
mean(edge_num_xor)
mean(diameter)
mean(density)
mean(average_pl)
mean(kl)
mean(avg_f1)