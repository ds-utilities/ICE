function [Y_predicteds] = f_predict_randForest(X, m, numTrees)
% Make predictions by the simple single instance and single-classifier
%  prediction mode.
% m is the model 

% Y_predicteds = zeros(size(X,1), 1);
  
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
f_gen_arff_forClass(fn_arff, X);
        

weka_bc = 'weka.classifiers.trees.RandomForest ';
% weka_bc = [weka_bc, '-l ', int2str(numTrees), '-K 0 -S 1 '];

        
weka_classifier = weka_bc; % this time, we won't use MIL, only SIL for now
fn_pred = [path_arff, id, '.pred'];
weka_command =['java -cp ', path_weka, ' ', weka_classifier, ...
    ' -T ', fn_arff, ' -l ', m, ' -p 0 > ', fn_pred];
system(weka_command);

Y_predicteds = f_read_wekaOut_P0(fn_pred);
% delete all the internal files
system(['rm -f ', fn_arff]);
system(['rm -f ', m]);
system(['rm -f ', fn_pred]);
    
   

end