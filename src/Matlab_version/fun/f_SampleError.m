
% by Will Dwinnell
%
% Last modified: Apr-17-2008
%
% Error = SampleError(Predicted,Actual,ErrorType)
%
% Error      = Calculated error measure
% Predicted  = Predicted values (column vector)
% Actual     = Target values (column vector)
% ErrorType:
%   'L-1'
%   'L-2'
%   'L-4'
%   'L-16'
%   'L-Infinity'
%   'RMS'
%   'AUC' (requires tiedrank() from Statistics Toolbox)
%   'Bias'
%   'Conditional Entropy'
%   'Cross-Entropy' (assumes 0/1 actuals)
%   'F-Measure'
%   'Informational Loss'
%   'MAPE'
%   'Median Squared Error'
%   'Worst 10%'
%   'Worst 20%'
%

% % test AUROC functions
% % true score is 0.568;
% x=a;
% y=[ones(length(x)/2,1); zeros(length(x)/2,1)];
% %auc = colAUC(x, y),
% auc = SampleError(x, y, 'AUC'),
% % ans =
% % 
% %     0.5679
% % SampleError is great!



function Error = f_SampleError(Predicted,Actual,ErrorType)

switch upper(ErrorType)
    case {'L-1', 'L1', 'LAD', 'LAE', 'MAE', 'ABSOLUTE'}
        Error = mean(abs(Predicted - Actual));
 
    case {'L-2', 'L2', 'MSE', 'LSE'}
        Error = mean((Predicted - Actual) .^ 2);

    case {'L-4', 'L4'}
        Error = mean((Predicted - Actual) .^ 4);
 
    case {'L-16', 'L16'}
        Error = mean((Predicted - Actual) .^ 16);

    case {'L-INFINITY', 'LINFINITY', 'MAXIMUM', 'CITYBLOCK', ...
            'MANHATTAN', 'TAXICAB', 'CHEBYSHEV', 'MINIMAX'}
        Error = max(abs(Predicted - Actual));

    case {'RMS', 'RMSE'}
        Error = sqrt(mean((Predicted - Actual) .^ 2));

    case {'AUC', 'AUROC'}
        % Count observations by class
        nTarget     = sum(double(Actual == 1));
        nBackground = sum(double(Actual == 0));

        % Rank data
        R = tiedrank(Predicted);  % 'tiedrank' from Statistics Toolbox

        % Calculate AUC
        Error = (sum(R(Actual == 1)) - (nTarget^2 + nTarget)/2) / (nTarget * nBackground);

    case {'BIAS'}
        Error = mean(Predicted - Actual);

    case {'CONDITIONAL ENTROPY', 'RESIDUAL ENTROPY'}
        Error = ConditionalEntropy(Actual,Predicted);

    % Note: errors of 1.0 blow up        
    case {'CROSS-ENTROPY', 'CROSSENTROPY', 'INFORMATIONALLOSS', 'INFORMATIONAL LOSS', 'MXE'}
        Error = mean(-log2([Predicted(Actual == 1); 1 - Predicted(Actual == 0)]));
        
    case {'F-MEASURE', 'F MEASURE'}
        TwoTP = 2 * sum(double( (Predicted == 1) && (Actual == 1) ));
        FP = sum(double( (Predicted == 1) && (Actual == 0) ));
        FN = sum(double( (Predicted == 0) && (Actual == 1) ));
        Error = TwoTP / (TwoTP + FP + FN);
        clear TwoTP FP FN;

    % Watch out for actuals equal to zero!
    case {'MAPE', 'RAE', 'RELATIVE'}
        Error = mean(abs((Predicted - Actual) ./ Actual));
        
    case {'MEDIAN SQUARED ERROR', 'MEDIAN SQUARE ERROR'}
        Error = median((Predicted - Actual) .^ 2);

    case {'WORST 10%'}
        [PredictedSorted I] = sort(Predicted);
        Error = sum(Actual(I(round(0.9 * length(Predicted)):end))) / sum(Actual);

    case {'WORST 20%'}
        [PredictedSorted I] = sort(Predicted);
        Error = sum(Actual(I(round(0.8 * length(Predicted)):end))) / sum(Actual);
end


% EOF

