function [ pred ] = f_main_RF_weka_file(train_arff, test_arff, nTrees)
%
% Example:
% y_preds = f_main_RF_kFold('/Users/zhengao/Desktop/arffs/', 'data_nki_NB_62fe', 10, [50 100 200]);

if ~ismac() % if not on my mac, On CBI
    path_weka = '/home/zhen.gao/bio/weka.jar ';
    path_tmp = './';
else % if on my Macbook
    path_weka = '/Users/zhengao/Dropbox/bio/weka.jar ';
    path_tmp = ['/Volumes/Macintosh_HD/Users/zhengao/bio/3_ensembF/',...
        '2_ref/Jamiul_code/ZA/'];
end
rand_str = [datestr(now, 'ddHHMMSS'),int2str(randi(10000)) ];
path_tmp = [path_tmp, rand_str, '/'];
%%

weka_classifier = ['weka.classifiers.trees.RandomForest -I ',int2str(nTrees)...
    ,' '];

% train and test
weka_command =['java -cp ', path_weka, ' ', weka_classifier, ...
    ' -t ', train_arff, ' -T ', test_arff, ' -p 0 > ', path_tmp, 'tmp.txt'];

system(weka_command);
% read result
pred = f_read_wekaOut_P0([path_tmp, 'tmp.txt']);

% Delete the arff file to save space.
system(['rm -f ', path_tmp, 'tmp.txt']);
system(['rm -rf ', path_tmp]);



end

