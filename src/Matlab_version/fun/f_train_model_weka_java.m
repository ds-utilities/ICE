function [model] = f_train_model_weka_java(X,y,model_type)
% train model using Java version WEKA to matlab.

% -------------- This can be done in a Matlab startup file ----------------
% adding Weka Jar file and the matlab2weka Jar file
% if ~ismac() % if not on my mac, On CBI
%     path_weka = '/home/zhen.gao/bio/weka.jar';
%     path_lib = '/home/zhen.gao/MATLAB_lab/';
% else % if on my Macbook
%     path_weka = '/Users/zhengao/Dropbox/bio/weka.jar ';
%     path_lib = '/Users/zhengao/Dropbox/mylab/MATLAB_lab/';
% end
% path_matlab2weka = [path_lib, 'matlab2weka/matlab2weka/matlab2weka.jar'];
% javaaddpath(path_weka);
% javaaddpath(path_matlab2weka);
% -------------------------------------------------------------------------


% ----------------------- Determine WEKA path -----------------------------
import matlab2weka.*;
% if ~ismac() % if not on my mac, On CBI
%     %path_proj = '/home/zhen.gao/bio/3_ensembF/';
%     path_weka = '/home/zhen.gao/bio/weka.jar ';
% else % if on my Macbook
%     %path_proj = '/Volumes/Macintosh_HD/Users/zhengao/bio/3_ensembF/';
%     path_weka = '/Users/zhengao/Dropbox/bio/weka.jar ';
% end

% ---------------- Generate Java_WEKA 'instances' object. -----------------
id = [datestr(now,'dd-mm-yyyy_HH:MM:SS_FFF'), '_rand', int2str(randi(10000))];
X_weka = f_gen_wekaObj_forClass(X, y, id);


% ---------------------- Set WEKA model parameters ------------------------
switch model_type
case 'svm'
    import weka.classifiers.functions.SMO.*;
    import weka.classifiers.functions.supportVector.Puk.*;
	%create an java object
    model = weka.classifiers.functions.SMO();
    model.setC(1.0);
    model.setEpsilon(1.0E-12);
    %model.setNumFolds(-1); % No need. It should be default
    %model.setRandomSeed(1); % No need. It should be default
    %model.setToleranceParameter(0.001); % default
    
    % I just wanto to use the dafault parameter first.
%     Kernel = weka.classifiers.functions.supportVector.Puk();
%     Kernel.buildKernel(X_weka);
%     model.setKernel(Kernel);
    
case 'j48'
    import weka.classifiers.trees.J48.*;
	%create an java object
    model = weka.classifiers.trees.J48();
	%defining parameters
    %model.setConfidenceFactor(0.25); %Set the value of CF.
    %model.setMinNumObj(2); %Set the value of minNumObj.
    %model.setNumFolds(-1);
    %model.setSeed(1);
    
case 'randForest'
    import weka.classifiers.trees.RandomForest.*;
    import weka.classifiers.meta.Bagging.*;
	%create an java object
    model = weka.classifiers.trees.RandomForest();
	%defining parameters
    model.setMaxDepth(0); %Set the maximum depth of the tree, 0 for unlimited.
    model.setNumFeatures(0); %Set the number of features to use in random selection.
    model.setNumTrees(100); %Set the value of numTrees.
    model.setSeed(1);
    
case 'NaiveBayes'
    import weka.classifiers.bayes.NaiveBayes.*;
	%create an java object
    model = weka.classifiers.bayes.NaiveBayes();
	%defining parameters
    
case 'cart'
    import weka.classifiers.trees.SimpleCart.*;
	%create an java object
    model = weka.classifiers.trees.SimpleCart();
	%defining parameters
    
case 'logistic'
    import weka.classifiers.functions.Logistic.*;
	%create an java object
    model = weka.classifiers.functions.Logistic();
	%defining parameters
    
otherwise
    fprintf('Unknown learner \n');
end

% ----------------------- train the classifier ----------------------------
model.buildClassifier(X_weka); 

end





