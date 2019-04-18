function [Y_predicteds] = f_predict_weka_arff(X, m, model_type)
% Using weka arff file as intermidiate layer.
% m is the model 

    if ~ismac() % if not on my mac, On CBI
        % path_proj = '/home/zhen.gao/bio/3_ensembF/';
        path_proj_scratch = '/scratch/zhen.gao/bio/3_ensembF/';
        path_model = [path_proj_scratch, '1_data/2_model_pool/'];
        path_arff = [path_proj_scratch, '1_data/3_arff_pool/'];
        path_weka = '/home/zhen.gao/bio/weka.jar ';
    else % if on my Macbook
        path_proj = '/Volumes/Macintosh_HD/Users/zhengao/bio/3_ensembF/';
        path_model = [path_proj, '1_data/2_model_pool/'];
        path_arff = [path_proj, '1_data/3_arff_pool/'];
        path_weka = '/Users/zhengao/Dropbox/bio/weka.jar ';
    end
    id = [datestr(now,'dd-mm-yyyy_HH:MM:SS_FFF'), '_rand', int2str(randi(10000))];
    fn_arff = [path_arff, id, '.arff'];
    f_gen_arff_forClass(fn_arff, X);
        
    switch model_type
    case 'svm'
        weka_bc = 'weka.classifiers.functions.SMO ';
    case 'j48'
        weka_bc = 'weka.classifiers.trees.J48 ';
    case 'randForest'
        weka_bc = 'weka.classifiers.trees.RandomForest ';
    case 'NaiveBayes'
        weka_bc = 'weka.classifiers.bayes.NaiveBayes ';
    case 'cart'
        weka_bc = 'weka.classifiers.trees.SimpleCart ';
    case 'logistic'
        % weka base classifier
        weka_bc = 'weka.classifiers.functions.Logistic ';
    otherwise
        fprintf('Unknown learner \n');
        
    end
    
    weka_classifier = weka_bc; % this time, we won't use MIL, only SIL for now
    fn_pred = [path_arff, id, '.pred'];
    weka_command =['java -cp ', path_weka, ' ', weka_classifier, ...
        ' -T ', fn_arff, ' -l ', m, ' -p 0 > ', fn_pred];
    system(weka_command);

    Y_predicteds = f_read_wekaOut_P0(fn_pred);
    % delete all the internal files
    system(['rm -f ', fn_arff]);
    %system(['rm -f ', m]);
    system(['rm -f ', fn_pred]);
    
end