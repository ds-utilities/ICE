function [avg_aucs,ix_max, x_wins, auc_gains] = ...
    f_AB_auto_v1(name, val_mat_eucli, val_mat_cosine, val_mat_corr, f)
% 
if nargin<5
    f = fopen('/Volumes/Macintosh_HD/Users/zhengao/bio/3_ensembF/2_ref/Jamiul_code/ZA/tmp.txt', 'w');
end

sim_types = {'euclidean','cosine','correlation'};
sim_types2 = {'euclidean distance','cosine similarity','correlation'};

fprintf(f,'=========================================================================\n');
fprintf(f,'%s\n', name);
fprintf(f,'----------------------------\n');
%% print the AB validation matrices
fprintf(f,'    val_mat\n');
fprintf(f,'\t\teuclidean\t\t\t\t  cosine\t\t\t\t  correlation\n');

fprintf(f,'A\t\t');
fprintf(f,'%.2f\t%.2f\t%.2f\t  ', val_mat_eucli(1,1),val_mat_eucli(1,2),val_mat_eucli(1,3));
fprintf(f,'%.2f\t%.2f\t%.2f\t  ', val_mat_cosine(1,1),val_mat_cosine(1,2),val_mat_cosine(1,3));
fprintf(f,'%.2f\t%.2f\t%.2f\n', val_mat_corr(1,1),val_mat_corr(1,2),val_mat_corr(1,3));

fprintf(f,'B\t\t');
fprintf(f,'%.2f\t%.2f\t%.2f\t  ', val_mat_eucli(2,1),val_mat_eucli(2,2),val_mat_eucli(2,3));
fprintf(f,'%.2f\t%.2f\t%.2f\t  ', val_mat_cosine(2,1),val_mat_cosine(2,2),val_mat_cosine(2,3));
fprintf(f,'%.2f\t%.2f\t%.2f\n', val_mat_corr(2,1),val_mat_corr(2,2),val_mat_corr(2,3));

fprintf(f,'C\t\t');
fprintf(f,'%.2f\t%.2f\t%.2f\t  ', val_mat_eucli(3,1),val_mat_eucli(3,2),val_mat_eucli(3,3));
fprintf(f,'%.2f\t%.2f\t%.2f\t  ', val_mat_cosine(3,1),val_mat_cosine(3,2),val_mat_cosine(3,3));
fprintf(f,'%.2f\t%.2f\t%.2f\n', val_mat_corr(3,1),val_mat_corr(3,2),val_mat_corr(3,3));

fprintf(f,'Whole\t');
fprintf(f,'%.2f\t%.2f\t%.2f\t  ', val_mat_eucli(4,1),val_mat_eucli(4,2),val_mat_eucli(4,3));
fprintf(f,'%.2f\t%.2f\t%.2f\t  ', val_mat_cosine(4,1),val_mat_cosine(4,2),val_mat_cosine(4,3));
fprintf(f,'%.2f\t%.2f\t%.2f\n', val_mat_corr(4,1),val_mat_corr(4,2),val_mat_corr(4,3));
fprintf(f,'\n');

fprintf(f,'------------------------------------------------------------------\n');
%% choose one from euclidean distance, cosine and correlation
% % mean AB
% average AUC on whole data.
% tmp = val_mat_eucli([1,2,4],1:2);
tmp = val_mat_eucli(4,1:2);
mean_eu = nanmean(tmp(:));
tmp = val_mat_cosine(4,1:2);
mean_cosine = nanmean(tmp(:));
tmp = val_mat_corr(4,1:2);
mean_corr = nanmean(tmp(:));
% fprintf(f,'\tAverage AUC on A B\n');
fprintf(f,'\tAverage AUC on whole data\n');
fprintf(f,'\teuclid\t  cosine    correlation\n');
fprintf(f,'\t%.2f\t  %.2f\t    %.2f\n', mean_eu, mean_cosine, mean_corr);

tmp = [mean_eu; mean_cosine; mean_corr];
avg_aucs = tmp';
ix_max = find(tmp == nanmax(tmp)); ix_max = ix_max(1);
fprintf(f,'    --> Choose to use %s:\n', sim_types2{ix_max} );
%%
if ix_max == 1
    a = val_mat_eucli;
elseif ix_max == 2
    a = val_mat_cosine;
elseif ix_max == 3
    a = val_mat_corr;
end
%
fprintf(f,'\n');
fprintf(f,'    val_mat\n');
fprintf(f,'\t--> using x to predict y\n');
fprintf(f,'\t\t\tA\t\tB\t\tC\n');
fprintf(f,'\tA\t\t');
fprintf(f,'%.2f\t%.2f\t%.2f\n', a(1,1),a(1,2),a(1,3));
fprintf(f,'\tB\t\t');
fprintf(f,'%.2f\t%.2f\t%.2f\n', a(2,1),a(2,2),a(2,3));
fprintf(f,'\tC\t\t');
fprintf(f,'%.2f\t%.2f\t%.2f\n', a(3,1),a(3,2),a(3,3));
fprintf(f,'\tWhole\t');
fprintf(f,'%.2f\t%.2f\t%.2f\n', a(4,1),a(4,2),a(4,3));
fprintf(f,'\n');

fprintf(f,'------------------------------------------------------------------\n');
%%
%
n_self_wins = 0;
n_other_wins = 0;
n_whole_wins = 0;

auc_gain_self = 0;
auc_gain_other = 0;
auc_gain_whole = 0;

strs1 = {'AA    '; 'BA    '; 'wholeA' };
tmp = a([1,2,4], 1);
[sorted, ix] = sort(tmp, 'descend');
fprintf(f,'\t%s   >    %s   >  %s\n', strs1{ix(1)}, strs1{ix(2)}, strs1{ix(3)} );
fprintf(f,'\t%.2f\t\t  %.2f\t\t  %.2f\n', sorted(1), sorted(2), sorted(3) );

% -----------------------------------------------------
auc_gain_self = auc_gain_self + tmp(1).*2 - tmp(2) - tmp(3);
auc_gain_other = auc_gain_other + tmp(2).*2 - tmp(1) - tmp(3);
auc_gain_whole = auc_gain_whole + tmp(3).*2 - tmp(1) - tmp(2);
if tmp(1) == max(tmp)
    n_self_wins = n_self_wins + 1;
end
if tmp(2) == max(tmp)
    n_other_wins = n_other_wins + 1;
end
if tmp(3) == max(tmp)
    n_whole_wins = n_whole_wins + 1;
end
% -----------------------------------------------------


if ix(1) == 1
    fprintf(f,'    --> use PCC or model build on A to predict instances in cluster A\n');
elseif ix(1) == 2
    fprintf(f,'    --> build model on B to predict instances in cluster A\n');
elseif ix(1) == 3
    fprintf(f,'    --> build model on whole data to predict instances in cluster A\n');
end

% -----------------------------------------------------
% -----------------------------------------------------
% -----------------------------------------------------
fprintf(f,'\n');
%
strs1 = {'AB    '; 'BB    '; 'wholeB' };
tmp = a([1,2,4], 2);
[sorted, ix] = sort(tmp, 'descend');
fprintf(f,'\t%s   >    %s   >  %s\n', strs1{ix(1)}, strs1{ix(2)}, strs1{ix(3)} );
fprintf(f,'\t%.2f\t\t  %.2f\t\t  %.2f\n', sorted(1), sorted(2), sorted(3) );

% -----------------------------------------------------
auc_gain_self = auc_gain_self + tmp(2).*2 - tmp(1) - tmp(3);
auc_gain_other = auc_gain_other + tmp(1).*2 - tmp(2) - tmp(3);
auc_gain_whole = auc_gain_whole + tmp(3).*2 - tmp(1) - tmp(2);
if tmp(2) == max(tmp)
    n_self_wins = n_self_wins + 1;
end
if tmp(1) == max(tmp)
    n_other_wins = n_other_wins + 1;
end
if tmp(3) == max(tmp)
    n_whole_wins = n_whole_wins + 1;
end
% -----------------------------------------------------
x_wins = [n_self_wins, n_other_wins, n_whole_wins];
auc_gains = [auc_gain_self, auc_gain_other, auc_gain_whole];
% -----------------------------------------------------


if ix(1) == 2
    fprintf(f,'    --> use PCC or model build on B to predict instances in cluster B\n');
elseif ix(1) == 1
    fprintf(f,'    --> build model on A to predict instances in cluster B\n');
elseif ix(1) == 3
    fprintf(f,'    --> build model on whole data to predict instances in cluster B\n');
end

fprintf(f,'\n');
%

fprintf(f,'------------------------------------------------------------------\n');


end

