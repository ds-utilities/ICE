function [prob_p, prob_n ] = f_NB_pred_2(c_p, c_n, cnt_y_p, cnt_y_n, Xi)
% This version just ignores the leave-one-out corss-validation. Assume the
%  counting table model has been adjusted, and only predict the y label. 


% Xi is one instance.
% [c_p, c_n, y,] is the counting tables model, build without the curent
%  instance.

% here assuming the values of Xi is converted from 
%   [-2, -1, 0, 1, 2] to [1,2,3,4,5]

% f_NB_pred_try_to_boost() trys to boost the speed, but failed. So I just
%  delete that part of code to make the function clean.


% % ---------------------- Adjust the counting tables: ----------------------
% if yi
%     for j=1:length(Xi)
%         c_p(Xi(j), j) = c_p(Xi(j), j) -1;
%     end
% else
%     for j=1:length(Xi)        
%         c_n(Xi(j), j) = c_n(Xi(j), j) -1;
%     end
%     %c_n(idx) = c_n(idx) -1;
% end
% % ---------------------- Adjust y ----------------------
% %     if yi == 1
% %         cnt_y_p = cnt_y_p-1;
% %     elseif yi == 0
% %         cnt_y_n = cnt_y_n-1;
% %     end
% % boost speed:
% % yi needs to be logical rather than number
% cnt_y_p = cnt_y_p -  yi;
% cnt_y_n = cnt_y_n - ~yi;
% OK, the result is correct.


% total number of instances
ni = cnt_y_p + cnt_y_n;

% ---------------------- Compute the posterior prob ----------------------
% prob_p = (cnt_y_p./ni) .* ni;
% prob_n = (cnt_y_n./ni) .* ni;


likehood_p = (cnt_y_p./ni) ; % P(positive)
likehood_n = (cnt_y_n./ni) ; % P(negative)
%  cnt_y_p,
%  cnt_y_n,

for j=1:length(Xi)
%      size(c_p),
%      [Xi(j), j],
    
    likehood_p = likehood_p .* (c_p(Xi(j), j) ./ (cnt_y_p+1) );

    likehood_n = likehood_n .* (c_n(Xi(j), j) ./ (cnt_y_n+1) );
end
% likehood_p,
% likehood_n,
% find(c_p <= 0 ),
% find(c_n(:)<= 0 ),

prob_p = likehood_p ./ (likehood_p + likehood_n);
prob_n = likehood_n ./ (likehood_p + likehood_n);


end

