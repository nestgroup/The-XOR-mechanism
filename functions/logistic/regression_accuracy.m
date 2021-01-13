function acc = regression_accuracy(y_pred,y_test)



acc = sum(y_pred==y_test)/length(y_test);
if acc<0.5
    acc = 1-0.5; % flip the labels
end

end