function cutoffs = getCutoffs(rwmatrix, avgNeighborsSize)
% 
% rwmatrix,
[a, ~] = sort(rwmatrix(:), 'descend');
len = size(rwmatrix, 1);
cutoffs = [];
for i=1:length(avgNeighborsSize)
    all_neibs = avgNeighborsSize(i) .* len;
    ct = a(all_neibs + 1);
    cutoffs = [cutoffs; ct];
    
end


end

