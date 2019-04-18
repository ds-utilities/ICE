function [ new ] = f_my_quantileNorm(templete, target)
% templete is the standard, change the target to the values in the
% templete. target may have a very different range than the templete. 

[~, ix_ta] = sort(target);
[~, ix_te] = sort(templete);

target(ix_ta) = templete(ix_te);
new = target;

end




