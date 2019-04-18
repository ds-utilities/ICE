function auc = f_eval_randForest(fn_arff, numTrees, k_fold)

% [model] = f_train_model_randForest(X_train,y_train, numTrees);
% [Y_predicteds] = f_predict_randForest(X_test, model, numTrees);



% model_type = 'randForest';
if ~ismac() % if not on my mac, On CBI
    path_proj = '/home/zhen.gao/bio/3_ensembF/';
    path_arff = [path_proj, '1_data/3_arff_pool/'];
    path_weka = '/home/zhen.gao/bio/weka.jar ';
else % if on my Macbook
    path_proj = '/Volumes/Macintosh_HD/Users/zhengao/bio/3_ensembF/';
    path_arff = [path_proj, '1_data/3_arff_pool/'];
    path_weka = '/Users/zhengao/Dropbox/bio/weka.jar ';
end

id = [datestr(now,'dd-mm-yyyy_HH:MM:SS_FFF'), '_rand', int2str(randi(10000))];
fn_res = [path_arff, id, '.res'];

% using WEKA
% 1. write the X and Y data to arff file
% f_gen_arff_forClass(fn_arff, X, y);

weka_bc = 'weka.classifiers.trees.RandomForest';
weka_bc = [weka_bc, ' -I ', int2str(numTrees), ' '];

weka_classifier = weka_bc; % this time, we won't use MIL, only SIL for now
weka_command =['java -cp ', path_weka, ' ', weka_classifier, ...
    ' -t ', fn_arff, '  -x ',int2str(k_fold),' -o -i > ', fn_res];

system(weka_command);


auc = f_read_wekaOut_auc_CV(fn_res);
system(['rm -f ', fn_res]);




end