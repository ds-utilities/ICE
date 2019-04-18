function [y_pred, auc, kap] = f_weka_RF_tr_te(X_tr, y_tr, X_te, y_te, I)

if nargin < 5
    I = 100;
end
rand_str = [datestr(now, 'ddHHMMSS'),int2str(randi(10000000000)) ];
if ispc
    path_tmp = ['C:\Users\gozhe\OneDrive\Desktop\APPs\tmp\'];
    path_tmp = [path_tmp, rand_str, '\'];
    path_weka = 'C:\Users\gozhe\Dropbox\bio\weka.jar ';  
    pa_java = 'java -d64 -Xms512m -Xmx20g ';
    del_temp_files_comm = ['rd /s /q ', path_tmp];
elseif ismac
    path_tmp = ['/Users/zhengao/Desktop/tmp/'];
    path_tmp = [path_tmp, rand_str, '/'];
    path_weka = '/Users/zhengao/Dropbox/bio/weka.jar ';
    pa_java = 'java -d64 -Xms512m -Xmx20g ';
    del_temp_files_comm = ['rm -rf ', path_tmp];
end

mkdir(path_tmp);

%% train data to arff file
arff_tr = [path_tmp, 'tr.arff'];
f_gen_arff_forClass(arff_tr, X_tr, y_tr);

%% test data to arff file
arff_te = [path_tmp, 'te.arff'];
% f_gen_arff_forClass(arff_te, X_te);
f_gen_arff_forClass(arff_te, X_te, y_te);
% since in another version, I need to read the y label from the testing
%  arff file, so I need to add the real y label in the arff testing file. 
%  However, it is still real testing, there is no information leaking /
%  cheating.

%% weka computing

% weka_classifier = ['weka.classifiers.functions.SMO ' ];
% weka_classifier = ['weka.classifiers.trees.RandomForest -I 100 ' ];
% weka_classifier = ['weka.classifiers.trees.RandomForest -I 100 ' ];
% weka.classifiers.trees.RandomForest -I 50 -K 0 -S 1
weka_classifier = ['weka.classifiers.trees.RandomForest -I ',int2str(I),' -S ',...
    int2str(randi(100000)), ' ' ];


% train and test
fn_pred = [path_tmp, 'tmp.txt'];
weka_command =[pa_java, '  -cp ', path_weka, ' ', weka_classifier, ...
    ' -t ', arff_tr, ' -T ', arff_te, ' -p 0 > ', fn_pred];
% weka_command,

system(weka_command);
% read result
y_pred = f_read_wekaOut_P0(fn_pred);

%% Delete the tmp files to save space.
% system(['rm -f ', fn_pred]);
% system(['rm -f ', arff_tr]);
% system(['rm -f ', arff_te]);
% system(['rm -rf ', path_tmp]);

system(del_temp_files_comm);

%% compute AUC
%try
  auc = f_SampleError(y_pred, y_te, 'AUC');
%catch
%  auc = NaN;    
%end
% kap = f_my_01_kappa(y_pred, y_te);
kap = NaN;


end






