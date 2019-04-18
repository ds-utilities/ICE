function f_gen_arff_forClass_2(fn, head, X, y)
% Revised from f_gen_arff_forClass(). This version takes a input header.
%  Better speed-wise, but need to have the head information ahead. 
% Generate WEKA arff file for classification

f = fopen(fn, 'wt');
len = size(X, 2);
%
for i=1:length(head)
    fprintf(f, '%s\n', head{i});
end

%
nr = size(X,1);
for i=1:nr;
    
    for j=1:len;
        %fprintf(f, '%0.3f, ', X(i, j));
        fprintf(f, '%0.4f, ', X(i, j));
    end
    if nargin<4 % if y has not been specified, then print '?'
        fprintf(f, '? \n');
    else % if y has been specified as usual
        fprintf(f, '%d \n', y(i));
    end
    if rem(i, 50)==0
        fprintf('%d / %d \n', i, nr);
    end
end

fclose(f);


end

