function p = RWR(A, nSteps, laziness, p0)
% the random walk algorithm. 
% A is the input net matrix, with the diag to be 0. 
% nSteps: how many steps to walk
% laziness: the probablity to go back. 
% p0: the initial probability. usually it is a zero matrix with the diag to
%   be 1. 
%
% for example, A could be: 
%  A = [0,2,2,0,0,0,0;...
%      2,0,1,1,0,0,0;...
%      2,1,0,0,1,0,0;...
%      0,1,0,0,0,1,1;...
%      0,0,1,0,0,0,0;...
%      0,0,0,1,0,0,1;...
%      0,0,0,1,0,1,0]
% 
% if nSteps is 1000 and laziness is 0.3, p0 is default, the result is:
%    [0.449, 0.207, 0.220, 0.064, 0.154, 0.034, 0.034;...
%     0.207, 0.425, 0.167, 0.132, 0.117, 0.071, 0.071;...
%     0.220, 0.167, 0.463, 0.052, 0.324, 0.028, 0.028;...
%     0.048, 0.099, 0.039, 0.431, 0.027, 0.232, 0.232;...
%     0.038, 0.029, 0.081, 0.009, 0.356, 0.004, 0.004;...
%     0.017, 0.035, 0.014, 0.154, 0.009, 0.425, 0.203;...
%     0.017, 0.035, 0.014, 0.154, 0.009, 0.203, 0.425]
%
% Each column represents the propability for each node. each element in the
%  column means the probability to go to that node.
% This algorithm will converge. For example, for the above matrix, nSteps =
%  100, 1000 or 10000, will give the same result. 




n = length(A);
if (nargin < 4)
    % eye(n) works the same as diag(ones(n, 1))
    p0 = eye(n);
end

% In the example above, spdiags(sum(A)'.^(-1), 0, n, n) will be 
%     0.2500         0         0         0         0         0         0
%          0    0.2500         0         0         0         0         0
%          0         0    0.2500         0         0         0         0
%          0         0         0    0.3333         0         0         0
%          0         0         0         0    1.0000         0         0
%          0         0         0         0         0    0.5000         0
%          0         0         0         0         0         0    0.5000

% W will be:
%          0    0.5000    0.5000         0         0         0         0
%     0.5000         0    0.2500    0.3333         0         0         0
%     0.5000    0.2500         0         0    1.0000         0         0
%          0    0.2500         0         0         0    0.5000    0.5000
%          0         0    0.2500         0         0         0         0
%          0         0         0    0.3333         0         0    0.5000
%          0         0         0    0.3333         0    0.5000         0


W = A * spdiags(sum(A)'.^(-1), 0, n, n);
p = p0;
pl2norm = inf;
unchanged = 0;
for i = 1:nSteps
    
    if rem(i, 100)==0
        fprintf('      done rwr %d \n', i-1);
    end
    
    pnew = (1-laziness) .* W * p + laziness .* p0;
    l2norm = max(sqrt(sum((pnew - p).^2) ) );
    p = pnew;
    if (l2norm < eps)
        break;
    else
        if (l2norm == pl2norm)
            unchanged = unchanged +1;
            if (unchanged > 10)
                break;
            end
        else
            unchanged = 0;
            pl2norm = l2norm;
        end
    end
end