function X_weka = f_gen_wekaObj_forClass(X, y, id)
% Generate a java-WEKA object for classification
% id is the relation name / data name
% X_weka: is the Java_WEKA 'instances' object.

% convert the logical/number 0-1 classes to WEKA nominal classies.
import matlab2weka.*;

y_nom = cell(size(y));
% if there is no y specified, that means the data is used for testing
if numel(y) == 0
    %y_nom = repmat({'?'}, size(X, 1));
    % since there is java error if use ?, then I just add a dummy class
    %  label. 
    y_nom = repmat({'c0'}, size(X, 1),1);
    %y_nom,
else
    uy = unique(y);
    tmp_cell = cell(1,1);
    for i = 1:length(uy)
        tmp_cell{1,1} = strcat('c', num2str(i-1));
        y_nom(y == uy(i),:) = repmat(tmp_cell, sum(y == uy(i)), 1);
    end
    clear uy tmp_cell i
end

ftName = cell(1, size(X,2));
len = size(X, 2);
for i=1:len;
    ftName{1, i} = ['f',int2str(i)];
end
% y_nom,
%convert the training data to an Weka object

% id, ftName,
% X'
% size(X), size(y_nom)
convert2wekaObj = convert2weka(id, ftName, X', y_nom, true);
X_weka = convert2wekaObj.getInstances();
clear convert2wekaObj;


end