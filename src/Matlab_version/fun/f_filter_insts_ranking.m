function [tf, rnk] = f_filter_insts_ranking(y_pred, y, sm)
% Giving the predicted y, and the soft-margin, see which instances can be
%  corrected predicted by the model, in order to filter out the bad
%  instances when re-train the model, or to associate the model with
%  instances. 
% This function also gives the rank of the each model by the performance. 

% sm: soft margin, an integer number. 

% tf: indicates which are good instances (correct predicted instances) and 
%  will be re-train. 

% number of actual positives
n_ap = sum(y);
[~, ix] = sort(y_pred, 'descend');

% the soft margin limit:
sm_lim = min([n_ap; length(y)-n_ap ]);
if sm > sm_lim
    sm = sm_lim;
elseif sm < -sm_lim
    sm = -sm_lim;
end
% *: limit: [-sm_lim sm_lim]

% predicted positives:
pp = false(size(y));
pp(ix(1:n_ap+sm)) = true;
% fprintf('soft margin = %d \t auc = %f\n', sm, f_SampleError(pp, y, 'AUC'));

% predicted negatives:
pn = false(size(y));
pn(ix(n_ap+1-sm:end)) = true;

% correct predicted positive vector:
tf_cp = pp & y;

%  figure, imagesc([pp, y, tf_cp]);

% correct predicted negative vector:
tf_cn = pn & ~y;

tf = tf_cp | tf_cn;


% rank: small to big
[~,~,rnk_sm2bg] = unique(y_pred);
a = rnk_sm2bg;
% rank, big to small
rnk_bg2sm = a+length(a)-2.*a+1;

% rank, correct predicted positives
rnk_cp = rnk_bg2sm;
% rank, correct predicted negatives
rnk_cn = rnk_sm2bg;

% figure, imagesc([y_pred.*200 rnk_cp rnk_cn  y.*200]);

rnk_cp(~tf_cp) = nan;
rnk_cn(~tf_cn) = nan;

% figure, imagesc([y_pred.*200 rnk_cp rnk_cn  y.*200]);
% figure, subplot(1,2,1); hist(rnk_cp); subplot(1,2,2); hist(rnk_cn);
% sum(tf), sum(tf_cp), sum(tf_cn),
% sum(~isnan(rnk_cp)),  sum(~isnan(rnk_cn)),

rnk = rnk_cp;
rnk(~isnan(rnk_cn)) = rnk_cn(~isnan(rnk_cn));

end

