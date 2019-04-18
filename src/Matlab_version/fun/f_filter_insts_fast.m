function tf = f_filter_insts_fast(y_pred, y, sm)
% Giving the predicted y, and the soft-margin, see which instances can be
%  corrected predicted by the model, in order to filter out the bad
%  instances when re-train the model, or to associate the model with
%  instances. 
% This function also gives the rank of the each model by the performance. 

% sm: soft margin, an integer number.
%  When sm > 0, less stringent
%  When sm < 0, more stringent

% tf: indicates which are good instances (correct predicted instances) and 
%  will be re-train. 

% number of actual positives
n_ap = sum(y);
[~, ix] = sort(y_pred, 'descend');

% the soft margin limit:
% sm_lim = min([n_ap; length(y)-n_ap ]);
% if sm > sm_lim
%     sm = sm_lim;
% else % sm < -sm_lim
%     sm = -sm_lim;
% end

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


end

