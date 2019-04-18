function [pred] = f_main_PCC_rf(train_arff, test_arff, ...
    rw_netF_prefix, n_models, n_trees)
%
if nargin < 5
    n_trees = 100;
end


if ~ismac() % if not on my mac, On CBI
    pa = '';
else % if on my Macbook
    pa = ['/Volumes/Macintosh_HD/Users/zhengao/bio/3_ensembF/2_ref/',...
    'Jamiul_code/PCC_code_and_test_data/'];
end

java_class_name = 'PCC_rf';
command = ['java -cp ',pa,'weka.jar:',pa,'jmatio.jar:',pa,...
    ' ',java_class_name,'  ',train_arff, ' ', test_arff, ' ',...
    rw_netF_prefix, ' ', f_arr2str(n_models, '-'), ' ', int2str(n_trees)];

command,
[~, result] = system(command),

pred = load([test_arff(1:end-4), 'txt']);


end

