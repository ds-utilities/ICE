function [arr] = f_read_JamiulCode_Out(fn)
% Read Jamiul's code output, in the 'PCC' project.
% Path of example output files:
% /Volumes/Macintosh_HD/Users/zhengao/bio/3_ensembF/2_ref/Jamiul_code/
%  PCC_code_and_test_data/res/nki_10folds/RandomForest/

%fn,
f = fopen(fn, 'r');

% Go over 'Making models for paient: xx cutoff: 0.xxx'
l=fgetl(f);
while ~feof(f)
    if length(l) >= 7 && strcmp(l(1:6), 'Making')
        l=fgetl(f);
    else
        break;
    end
end

% Go over '0 0' to 'xx xx' etc.
% l=fgetl(f);
while ~feof(f)
    if length(l) > 20
        break;
    else
        l=fgetl(f);
    end
end
% l,

i=1;
while ~feof(f)
    b = str2num(l);
    l=fgetl(f);
    arr(i, 1:length(b)) = b;
    i=i+1;

end
b = str2num(l);
arr(i, 1:length(b)) = b;


fclose(f);


end

