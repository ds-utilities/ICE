function [model] = f_train_model(X, y, model_type, isWekaJava, useMATLAB)
% X: train data,
% Y: class labels
% model_type: classification algorithm.
% model: the trained model
%
% isWekaJava: using weka java or weka command-line and files. 1 for
%  weka-java, 0 for command-line + files.
%
% useMATLAB: use MATLAB learners 'linear regress' or 'svm' or use 
%  WEKA learners 'svm', 'random forest', etc..
%  0 for use WEKA, 1 for use MATLAB
%
% if useMATLAB < 5
%     useMATLAB = 0; % default using Weka learners.
% end
% if nargin < 4
%     % default using Weka command line and files.
%     isWekaJava = 0;
% end

if useMATLAB==1
    if strcmp(model_type, 'linear_regress')
        b = regress(y, [X, ones(size(y))]);
        model = b;
        % note: when using MATLAB linear regression, the model is just a number
        %  vector. When using weka, the model is a model file name.

    elseif strcmp(model_type, 'svm')
        % try MATLAB svm instead of WEKA to see if it is faster and usable.
        mdl = fitrsvm([X, ones(size(y))],  y+1-1);
        model = mdl;
    end
else 
    % using MATLAB only for linear regression, otherwise, using WEKA
    if isWekaJava == 1
        [model] = f_train_model_weka_java(X,y,model_type);
    else
        % use weka file
        [model] = f_train_model_weka_file(X,y,model_type);
    end
    
end

end

