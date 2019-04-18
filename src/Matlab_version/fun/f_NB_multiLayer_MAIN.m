function f_NB_multiLayer_MAIN( train_sets, test_sets, X, y, k_fold,  ...
                        idxs, modelss, rs, nRPs, auc_cts, layers, f)
% 



% layers = 1;

% % sim = IDFsim(X);
% sim = squareform(pdist(X, 'cosine')); sim = -sim;
% for repeat = 1:10    
% Generate a feature random re-sampling division:
% for r = [ 200 300 ]
% for r = [ 200 300 400 ]
for r = rs
    % for nRP = [1 5  round(2000./r) 100]
    %for nRP = [1  10 50 100 1000]
%     for nRP = [ 50 100 ]
    for nRP = nRPs
        groups = f_gen_grps(size(X,2), r, nRP);

        %for NN = [ 1 2 3 5 10 50  ];
%             for auc_ct = [ .75 .77 .8 .82 ] 
            for auc_ct = auc_cts
                
                y_preds = zeros(length(y), 1);
                
                alg = ['NB_METHODx_noReTrain_', 'rand', '_r', int2str(r),'_nRP', ...
                    int2str(nRP),  '_aucCT_',num2str(auc_ct), ...
                    '_', f_arr2str(layers), '_','NB',  '_cv', int2str(k_fold)];
                
                parfor fold = 1:length(test_sets)
                %for fold = 2                    
                    % ------------------------ prepare data -----------------------------
                    fprintf('fold = %d \n', fold);
                    %[ Y_predicted ] = f_pred_all_clustLayers_NB( train_sets, test_sets, ...
                    %    fold, X, y, idxs, modelss,  auc_ct, groups);
                    [ Y_predicted ] = f_pred_clustLayers_NB( train_sets, test_sets, ...
                        fold, X, y, idxs, modelss,  auc_ct, groups, layers);
                    
                    y_preds(fold) = Y_predicted;
                    
                end % fold
                
                % ------------------------ Compute AUC -----------------------------
                auc = f_SampleError(y_preds, y, 'AUC');
                
                tmpstr = [alg, ',',  int2str(r),',',  int2str(nRP),',', 'NN_NA' ,','...
                    f_arr2str(layers), ',',   num2str(auc_ct), ',',   'NB', ',', ...
                    int2str(k_fold),',',num2str(auc), '\n'];
                
                fprintf(f, tmpstr);
                
                
            end % auc_ct
            
        % end % NN
        
    end % nRP
end % r





end

