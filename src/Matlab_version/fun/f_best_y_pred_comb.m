function [aucBST,nMS1,nMS2,  aucBST1only,nMS1only,  aucBST2only,nMS2only]=...
    f_best_y_pred_comb(n_models_1, y_pred_1, n_models_2, y_pred_2, y, ...
    classifier_1_name, classifier_2_name)
% Find the best y_prediction combination of 2 different classifiers. Both 
%  classifiers are ensemble methods. Each classifier has used different
%  number of models. This algorithm finds the best number of models when
%  combine the 2 classifiers.
%
% n_models_1: a list of number of models for classifier 1
% y_pred_1: the corresponding y predictions using different # of models
% n_models_2: a list of number of models for classifier 2
% y_pred_2: the corresponding y predictions using different # of models
% y: the real class labels
%
% aucBST: The best combination AUC of the 2 classifiers
% nMS1: for the final best AUC, number of classifier 1 models
% nMS2: for the final best AUC, number of classifier 2 models
% aucBST1only: best AUC using classifier 1 only
% nMS1only: number of models used for the best classifier 1 AUC
% aucBST2only: best AUC using classifier 2 only
% nMS2only: number of models used for the best classifier 2 AUC

if nargin < 6
    classifier_1_name = 'Classifier 1';
    classifier_2_name = 'Classifier 2';
end


% -------------------------------------------------------------------------
% Compute the best AUC and corresponding #models using classifier 1 only
aucs1 = zeros(length(n_models_1), 1);
for i=1:length(n_models_1)
    aucs1(i) = f_SampleError(y_pred_1(:, i), y, 'AUC');
end
aucBST1only = max(aucs1);
ix = find(aucs1 == aucBST1only);
ix = ix( ceil(length(ix)/2) );
nMS1only = n_models_1(ix);

% -------------------------------------------------------------------------
% Compute the best AUC and corresponding #models using classifier 2 only
aucs2 = zeros(length(n_models_2), 1);
for i=1:length(n_models_2)
    aucs2(i) = f_SampleError(y_pred_2(:, i), y, 'AUC');
end
aucBST2only = max(aucs2);
ix = find(aucs2 == aucBST2only);
ix = ix(1);
nMS2only = n_models_2(ix);

% -------------------------------------------------------------------------
% Compute the best combination AUC
aucs_combine = zeros(length(n_models_1), length(n_models_2));
for i=1:length(n_models_1)
    for j=1:length(n_models_2)
        aucs_combine(i, j) = ...
            f_SampleError(mean([y_pred_1(:, i), y_pred_2(:, j)], 2  ), y, 'AUC');
    end
end
aucBST = max(aucs_combine(:));
[r, c] = find(aucs_combine == aucBST );
r = r(1);
c = c(1);

nMS1 = n_models_1(r);
nMS2 = n_models_2(c);

% -------------------------------------------------------------------------
fprintf('%s best AUC = %.3f,    using %d models\n', classifier_1_name,...
    aucBST1only, nMS1only);
fprintf('%s best AUC = %.3f,    using %d models\n', classifier_2_name,...
    aucBST2only, nMS2only);
fprintf('Hybrid best AUC = %.3f,    using %d %s models, %d %s models\n\n', ...
    aucBST, nMS1, classifier_1_name, nMS2, classifier_2_name );

end

