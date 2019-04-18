function [train_sets, test_sets] = ...
    f_arff_to_k_fold_arffs(fn, k_fold, out_path, prefix)
% 
% Pre-convert the whole arff file to 20 10-fold cross-validation arff
%  files. For each fold, there would be a training file with 90% instances
%  and a corresponding testing file with 10% instances. 

% n_lines = f_file_count_lines(fn);

[head, body] = f_read_arff_2_strs(fn);

n_insts = length(body);
[train_sets, test_sets] = f_StratifiedKFold(n_insts, k_fold);

for fold = 1:length(test_sets)
    % ------------------------ prepare data -----------------------------
    fold,
    train_loc = train_sets{fold, 1};
    test_loc = test_sets{fold, 1};
    
    fn_out = [out_path, prefix,'_fold_',int2str(fold), '_train', '.arff'];
    f = fopen(fn_out, 'wt');
    for i=1:length(head)
        fprintf(f, [head{i, 1}, '\n']);
    end
    for i=1:length(train_loc)
        fprintf(f, [body{train_loc(i), 1}, '\n']);
    end
    fclose(f);
    
    
    fn_out = [out_path, prefix,'_fold_',int2str(fold), '_test', '.arff'];
    f = fopen(fn_out, 'wt');
    for i=1:length(head)
        fprintf(f, [head{i, 1}, '\n']);
    end
    for i=1:length(test_loc)
        fprintf(f, [body{test_loc(i), 1}, '\n']);
    end
    fclose(f);
    
end

end

