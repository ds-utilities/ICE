function [Y_predicteds] = f_predict_weka_java(X, m)
% Using weka arff file as intermidiate layer.
% m is the model 

import matlab2weka.*;
    import weka.classifiers.functions.SMO.*;
    import weka.classifiers.functions.supportVector.Puk.*;
    import weka.classifiers.trees.J48.*;
    import weka.classifiers.trees.RandomForest.*;
    import weka.classifiers.meta.Bagging.*;
    import weka.classifiers.bayes.NaiveBayes.*;
    import weka.classifiers.functions.Logistic.*;

X_weka = f_gen_wekaObj_forClass(X, '', 'testing');
% X_weka,
Y_predicteds = true(size(X,1), 1);
for i=1:size(X,1)
    %X_weka.instance(i-1)
    %m.classifyInstance(X_weka.instance(i-1))
    %X_weka.instance(i-1).classAttribute
    %X_weka.instance(i-1).classAttribute.value(0)
    %X_weka.instance(i-1).classAttribute.value(1)
    pred = m.classifyInstance(X_weka.instance(i-1));
    % WEKA infor.:
    %                  Class
    % Attribute           c1      c0
    %                 (0.27)  (0.73)

    % Since I weka its internal sequence for the Classes are c1 then c0 
    %  order, I have to write as below: when 'pred == 0', the class label 
    %  is actually c1, if 'pred == 1', its class label is actually c0.
    if pred == 0
        Y_predicteds(i, 1) = true;
    else % else if pred == 1
        Y_predicteds(i, 1) = false;
    end
end

    
end