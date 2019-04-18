function [X, y] = f_read_arff_w_labels(fn)
% Read arff file with class labels
% Assume the last column is te class label, and all the features are
%  numerical.
[~, body] = f_read_arff_2_strs(fn);

n_col = length( str2num(body{1}) ); % becuase the last column is the label.
n_row = length(body);
X = zeros(n_row, n_col);

for i=1:length(body)
    X(i, :) = str2num(body{i});
end

y = X(:, end);
X = X(:, 1:end-1);

end

