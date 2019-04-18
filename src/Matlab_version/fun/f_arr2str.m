function str = f_arr2str(arr, deli)
% convert multiple cores to a string, 
% deli = ','; % or delimeter could be n, 
if nargin < 2
    deli = '_';
end

if length(arr) == 1
    str = num2str(arr);
else
    str = num2str(arr(1));
    for i=2:length(arr)
        str = [str, deli, num2str(arr(i))];
    end
end


end

