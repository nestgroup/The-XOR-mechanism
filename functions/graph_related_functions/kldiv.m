function kldist=kldiv(A,A_pvt)

N = size(A,1);

d1 = sum(A,2);

d2 = sum(A_pvt,2);


[count1,val1] = groupcounts(d1) ;
hist_d1 = zeros(N,1);
hist_d1(val1+1) = count1/N;


[count2,val2] = groupcounts(d2) ;
hist_d2 = zeros(N,1);
hist_d2(val2+1) = count2/N;


kl = hist_d2.*log(hist_d2./ (hist_d1+eps));
% kl(find(isinf(kl)==1))=0;
kl(find(isnan(kl)==1))=0;

kldist = sum(kl);

end