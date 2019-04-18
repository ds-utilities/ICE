function  f_delete_weka_models(models, model_type, big_model_type)
%
if strcmp(model_type, 'linear_regress')
    % do nothing
else
    if strcmp(big_model_type, 'pc')
        for i=1:length(models)
            ms_inst = models{i};
            for j = 1:length(ms_inst)
                m = ms_inst{j};                
                system(['rm -f ', m]);
            end
        end
        
    end
    
end



end

