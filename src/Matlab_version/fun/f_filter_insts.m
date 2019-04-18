function [tf] = f_filter_insts(y_pred, y, sm)
% Giving the predicted y, and the soft-margin, see which instances can be
%  corrected predicted by the model, in order to filter out the bad
%  instances when re-train the model, or to associate the model with
%  instances. 
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
% figure,
% subplot(1,3,1);
% imagesc([pp y]);

% predicted negatives:
pn = false(size(y));
pn(ix(n_ap+1-sm:end)) = true;
% subplot(1,3,2);
% imagesc([pn y]);

% correct predicted positive vector:
tf_cp = pp & y;

% correct predicted negative vector:
tf_cn = pn & ~y;

tf = tf_cp | tf_cn;



end

