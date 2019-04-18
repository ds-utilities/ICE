function arr = f_clustLoc_cell_to_arr(c)
% convert the cluster indicator format from cell to array

% find the length of the array
m=0;
for i=1:length(c)
    if max(c{i}) > m
        m = max(c{i});
    end
end
arr = zeros(m,1);
for i=1:length(c)
    arr(c{i}) = i;
end

end

