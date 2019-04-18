function [head, body] = f_read_arff_2_strs(fn)
%
head = {};
f = fopen(fn, 'r');

i=1;
l = fgetl(f);
% while ~feof(f)
while ~feof(f)
    head{i, 1} = l;
    i=i+1;
    if length(l)>=5 && strcmpi( l(1:5), '@data')
        break;
    else
        l = fgetl(f);
    end
    
end
body = {};
i=1;
while ~feof(f)
    l = fgetl(f);
    body{i, 1} = l;
    i=i+1;
end

fclose(f);



end

