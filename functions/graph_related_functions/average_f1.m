function avg_f1 = average_f1(T,P)
%{
T is the ground truth community
P is the predicted community
%}

[N,C1] = size(T); % N nodes, C1 communities
[N,C2] = size(P); % N nodes, C2 communities


%% part 1 (C1,C2)
f1_C1 = [];
for i = 1:C1
    f1_C2 = [];
    for j = 1:C2
        f1 = f1_cal(   T(:,i) , P(:,j)  );
        if isnan(f1)
            f1 = 0;
        end
        f1_C2 = [f1_C2 f1];
    end
    f1_C1 = [f1_C1 max(f1_C2)];
end

part1 = 1/(2*C1)*sum(f1_C1);
%% (C2,C1)
f1_C2 = [];
for i = 1:C2
    f1_C1 = [];
    for j = 1:C1
        f1 = f1_cal(P(:,i),T(:,j));
        if isnan(f1)
            f1 = 0;
        end
        f1_C1 = [f1_C1 f1];
    end
    f1_C2 = [f1_C2 max(f1_C1)];
end
part2 = 1/(2*C2)*sum(f1_C2);
avg_f1 = part1 +part2; 

        
end



function f1 = f1_cal(A,B)
    indxA = find(A==1);
    indxB = find(B==1);
    prec = length(intersect(indxA,indxB))/length(indxA);
    recall  = length(intersect(indxA,indxB))/length(indxB);
    f1 = 2*(prec*recall)/(prec+recall);
end



