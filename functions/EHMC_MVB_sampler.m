function B = EHMC_MVB_sampler(size1,size2,sf,epsilon,alpha, T)


%{
Implementation of the EHMC sampler for matrix-valued bernoulli
distributionbased using the Gaussian augmented-variable HMC sampler.

This implementation adapts the sampling scheme introduced by "Auxiliary-variable exact
Hamiltonian Monte Carlo samplers for binary distributions", which also
available at https://github.com/aripakman/hmc-binary
%}

%{
Input:
size1, size2:  the size of a binary matrix valued query
sf: sensitivity of binary- and matrix-valued query
epsilon: privacy budget
alpha: tunning parameter on the structural properties
T: travel time for the EHMC

Output:
B: a sample of binary noise that preserves epsilon-differential privacy for
the considered binary- and matrix-valued query
%}

%% generate PI


Theta =   speye(size2) * alpha * epsilon/(sf * sqrt(size2) ); % or choose any other orthonormal basis instead of the standard basis

J =   sparse(ones(size1))-speye(size1)  ;
Lij =  speye(size2) * (1-alpha) / ( size1*(size1-1)/2 ) * epsilon/(sf * sqrt(size2) ); % or choose any other orthonormal basis
PI = kron( speye(size1) , Theta) + kron(J,Lij);


%% sampling

S = EHMC_sampler(PI,T);

B = reshape(S,size1,size2);


end


%% log of the probability
function lp = logp(S,M)
lp = -S'*M*S  -S'*M*ones(length(S),1);
end


%% change of Hamiltonian based on the interactions between neighboring dipole specified by  PI
function lpc= logp_change(M,S,j)


idx = find(M(j,:)~=0);
idx( find(idx==j) ) = [];
pp = diag(M);
lpc_self = 2 * 2 * pp(j);
if isempty(idx) % dipole has no interacting neighbors
    lpc_nei = 0;
else
    pj = zeros(length(S),1);
    pj(j) = 1;
    lpc_nei =   2 * pj' * M * S;
end
lpc = lpc_self +  lpc_nei;




end



function S = EHMC_sampler(PI,T)

temp = 3*10^-5; % set temperature
L = 10;
M = - PI;
M = 1/temp * M/2;
d = size(M,1);
initial_X = abs(normrnd(0,1,d,1));
% initial_X =  normrnd(0,1,d,1);
log_likes = zeros(L,1);
energies  = zeros(L,1);
ll = logp(sign(initial_X),M);
log_likes(1) = ll;
Xs=NaN;
wall_hits =0;
wall_crosses =0;
nearzero= 10000*eps;
touched = zeros(d,int64(T/pi)*(L-1));
mts = zeros(d,int64(T/pi)*(L-1));
xx = zeros(d,1);
%% Sampling loop
last_X= initial_X;
Xs=zeros(d,L);
Xs(:,1)=initial_X;
i=2;
while (i <= L)
%     i
    stop=0;
    j=0;
    V= normrnd(0,1, d,1);   % initial velocity
    X = last_X;
    
    tt=0;                    % records how much time the particle already moved
    S=sign(X);

    while (1)
        
        a = V;
        b = X;
        phi = atan2(b,a);           % -pi < phi < +pi
        
        % find the first time constraint becomes zero
        
        wt1= -phi;                 % time at which coordinates hit the walls          % wt1 in [-pi/2, 3pi/3]
        wt1(phi>0) = pi -phi(phi>0);
        
        
        % if there was a previous reflection (j>0)
        % and there is a potential reflection at the sample plane
        % make sure that a new reflection at j is not found because of numerical error
        % c.f. Auxiliary-variable exact Hamiltonian Monte Carlo samplers for binary distributions
        
        if j>0
            tt1 = wt1(j);
            if abs(tt1) < nearzero || abs(tt1-2*pi)< nearzero
                wt1(j)=Inf;
            end
        end
        
        
        [mt, j] = min(wt1);
        
        tt=tt+mt;
        
        if tt>=T
            mt= mt-(tt-T);
            stop=1;
            
        else
            wall_hits = wall_hits + 1;
            
            
        end
        
        
        X = a*sin(mt) + b*cos(mt);
        V = a*cos(mt) - b*sin(mt);
        
        if stop
            break;
        end
        
        X(j) = 0;
        
        v2_new = V(j)^2 +  sign(V(j))*2*logp_change(M,S,j);
        if v2_new >0
            V(j) = sqrt(v2_new)* sign(V(j));
            S(j) = -S(j);
            wall_crosses = wall_crosses +1;
            
        else
            V(j) = -V(j);
        end
        
        
        
    end 
    
    
    Xs(:,i)=X;
    ll = logp(S,M);
    log_likes(i) = ll;
    energies(i) = 0.5*(X'*X + V'*V) -log_likes(i);
    last_X = X;
    i= i+1;
    
end %while (i <= L)

Sis= sign(Xs);

Zs   = (Sis+1)/2;

[~,idx] = min(energies(2:end));

S = Zs(:,idx);




end