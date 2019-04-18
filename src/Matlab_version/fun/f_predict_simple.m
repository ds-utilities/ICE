function [Y_predicteds] = f_predict_simple(X, m, model_type,...
    isWekaJava, useMATLAB)
% Make predictions by the simple single instance and single-classifier
%  prediction mode.
% m is the model 

% Y_predicteds = zeros(size(X,1), 1);

if useMATLAB == 1
    if strcmp(model_type, 'linear_regress')
        b=m(:);
        %size([X, ones(size(X,1),1)]),    
        Y_predicteds = [X, ones(size(X,1),1)] * b;    
        %if length(Y_predicteds) > 10
        %    % if the predicted y is long, then, map the result to [0 1]
        %    Y_predicteds=(Y_predicteds-min(Y_predicteds(:))) ./ ...
        %        ( max(Y_predicteds(:)) - min(Y_predicteds(:)) );
        %end
    elseif strcmp(model_type, 'svm')
        % try MATLAB svm intead of weka
        Y_predicteds = predict(m, [X, ones(size(X,1),1)]);
    end

else % if not using linear regression, then using WEKA    
    if isWekaJava == 1 % use WEKA-Java
        [Y_predicteds] = f_predict_weka_java(X, m);  
    else % use WEKA file
        [Y_predicteds] = f_predict_weka_arff(X, m, model_type);
    end
    
    
end

end