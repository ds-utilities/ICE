function [model] = f_train_model_weka_file(X,y,model_type)
% train model using commond line weka and arff files.

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
id = [datestr(now,'dd-mm-yyyy_HH:MM:SS_FFF'), '_rand', int2str(randi(100000))];
fn_arff = [path_arff, id, '.arff'];
% using WEKA
% 1. write the X and Y data to arff file
f_gen_arff_forClass(fn_arff, X, y);

fn_m = [path_model, model_type, '_', id, '.model'];
switch model_type
case 'svm'
    % When using weka, the model is a model file name, represented by a
    %  char string. 
    % fn_m: model file name 
    weka_bc = 'weka.classifiers.functions.SMO ';
case 'j48'
    weka_bc = 'weka.classifiers.trees.J48 ';
case 'randForest'
    weka_bc = 'weka.classifiers.trees.RandomForest -I 100 ';
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
weka_command =['java -cp ', path_weka, ' ', weka_classifier, ...
    ' -t ', fn_arff, ' -d ', fn_m, ' -no-cv'];
system(weka_command);
model = fn_m;
% Delete the arff file to save space.
system(['rm -f ', fn_arff]);


end

