function Y_predicteds = f_main_use_net_only(y_train, ...
    X_test, net, NN, train, test)
% Revised from f_main_pc_jamiul_RW_nets(), this time, use only the weighted
%  random-walk net or just the coeficient matrix. Predict the y label using
%  only the nestest neighbor's label. 


% ------------------------ Train Classifiers -----------------------------

% ------------------------- Make Predictions -----------------------------
net_sub = net(train, test);

[Y_predicteds] = f_predict_only_net(X_test, net_sub, NN, y_train);

end

