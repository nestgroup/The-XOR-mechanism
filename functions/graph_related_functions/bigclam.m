function  C_P = bigclam(A, K)



N = size(A,1);

[H,~] = nnmf(A, K);

beta = 0.5;

for i = 1:100

    H_new = max( H.*(1-beta + beta.* (A*H)./(H*H'*H)  ), 0);
    err(i,1) = norm( H_new -H, 'fro');
    err(i,2)  = norm( A - H_new*H_new', 'fro');
    if (  err(i,1) < 10^-3)
        H = H_new;
        break;
    end
    H = H_new;

end


[~,c] = max(H,[],2);
C_P =  zeros(N,K);
for ci = 1:K
    C_P(find(c==ci),ci)=1;
end


end