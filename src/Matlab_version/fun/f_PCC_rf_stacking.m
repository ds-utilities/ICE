function f_PCC_rf_stacking(train_arff, test_arff, ...
    rw_netF_prefix, n_model, n_trees, fn_out)
%
if nargin < 5
    n_trees = 100;
end

if ~ismac() % if not on my mac, On CBI
    pa = '';
    pa_tmp = '';
else % if on my Macbook
    pa = ['/Volumes/Macintosh_HD/Users/zhengao/bio/3_ensembF/2_ref/',...
    'Jamiul_code/PCC_code_and_test_data/'];
    pa_tmp = ['/Volumes/Macintosh_HD/Users/zhengao/bio/3_ensembF/2_ref/',...
        'Jamiul_code/ZA/'];
end

java_class_name = 'PCC_rf_3';
command = ['java -cp ',pa,'weka.jar:',pa,'jmatio.jar:',pa,...
    ' ',java_class_name,'  ',train_arff, ' ', test_arff, ' ',...
    rw_netF_prefix, ' ', int2str(n_model), ' ', int2str(n_trees), ...
    ' ', fn_out];

command,
[~, result] = system(command),



end

