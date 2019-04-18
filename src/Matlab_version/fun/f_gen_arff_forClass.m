function f_gen_arff_forClass(fn, X, y)
% Generate WEKA arff file for classification
% Don't need the arff head information like f_gen_arff_forClass_2(). 
% More convenient to use, simply need the data and file name. 


f = fopen(fn, 'wt');
%
fprintf(f, '@RELATION ');
% fprintf(f, fn);
fprintf(f, ' abc123 ');
fprintf(f, '\n');
 
%fprintf(fid1, '@ATTRIBUTE ');
%fprintf(fid1, 'row_labels ');
%fprintf(fid1, 'STRING\n');
%
len = size(X, 2);
for i=1:len;
    fprintf(f, '@ATTRIBUTE ');
    fprintf(f, ['f',int2str(i)]); % fx means the xth feature.
    fprintf(f, ' NUMERIC\n');
end

fprintf(f, '@ATTRIBUTE class {1, 0}');

fprintf(f, '\n');
fprintf(f, '\n');
fprintf(f, '@DATA\n');

%
nr = size(X,1);
for i=1:nr;
    
    for j=1:len;
        %fprintf(f, '%0.3f, ', X(i, j));
        fprintf(f, '%0.4f, ', X(i, j));
    end
    if nargin<3 % if y has not been specified, then print '?'
        fprintf(f, '? \n');
    else % if y has been specified as usual
        fprintf(f, '%d \n', y(i));
    end
    %if rem(i, 100)==0
    %    fprintf('%d / %d \n', i, nr);
    %end
end

fclose(f);


end

