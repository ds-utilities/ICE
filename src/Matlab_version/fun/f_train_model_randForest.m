function [model] = f_train_model_randForest(X,y, numTrees)
% X: train data,
% Y: class labels
% model_type: classification algorithm.
% model: the trained model
model_type = 'randForest';
if ~ismac() % if not on my mac, On CBI
    path_proj = '/home/zhen.gao/bio/3_ensembF';
    path_model = [path_proj, '1_data/2_model_pool/'];
    path_arff = [path_proj, '1_data/3_arff_pool/'];
    path_weka = '/home/zhen.gao/bio/weka.jar ';
else % if on my Macbook
    path_proj = '/Volumes/Macintosh_HD/Users/zhengao/bio/3_ensembF/';
    path_model = [path_proj, '1_data/2_model_pool/'];
    path_arff = [path_proj, '1_data/3_arff_pool/'];
    path_weka = '/Users/zhengao/Dropbox/bio/weka.jar ';
end

id = [datestr(now,'dd-mm-yyyy_HH:MM:SS_FFF'), '_rand', int2str(randi(10000))];
fn_arff = [path_arff, id, '.arff'];

% using WEKA
% 1. write the X and Y data to arff file
f_gen_arff_forClass(fn_arff, X, y);

fn_m = [path_model, model_type, '_', id, '.model'];

weka_bc = 'weka.classifiers.trees.RandomForest ';
weka_bc = [weka_bc, '-l ', int2str(numTrees), '-K 0 -S 1 '];

weka_classifier = weka_bc; % this time, we won't use MIL, only SIL for now
weka_command =['java -cp ', path_weka, ' ', weka_classifier, ...
    ' -t ', fn_arff, ' -d ', fn_m];
system(weka_command);
model = fn_m;
% Delete the arff file to save space.
system(['rm -f ', fn_arff]);

   
end

