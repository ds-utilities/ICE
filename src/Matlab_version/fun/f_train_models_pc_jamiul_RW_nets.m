function [classifiers, neighbor_infor] = f_train_models_pc_jamiul_RW_nets(...
    X_train, y_train, ix_train, model_type, err_lim, isWekaJava, useMATLAB)
% train models for the PC (personalized classifier) classifier

% if useMATLAB < 7
%     useMATLAB = 1; % default using MATLAB learners.
% end
% if nargin < 6
%     % default using Weka command line and files.
%     isWekaJava = 0;
% end


if ~ismac() % if not on my mac, On CBI
    path_proj = '/home/zhen.gao/bio/3_ensembF/';
else % if on my Macbook
    path_proj = '/Volumes/Macintosh_HD/Users/zhengao/bio/3_ensembF/';
end

rw_net_path = [path_proj, '1_data/1_nki/3_data_jamiul_RW_nets/'];

n_jamiul_nets = 5;
neighbor_infor = cell(1, n_jamiul_nets);
for i=1:n_jamiul_nets
    load([rw_net_path, 'RW_neighbors_',int2str(i),'.mat']);
    cur_net = neighbor;
    sub_net = cur_net( ix_train, ix_train);
    
    neighbor_infor{1, i} = sparse(sub_net);    
end


% =========================== Build Models ================================
% #model groups = #instances
classifiers = cell(1, size(X_train,1));
for node = 1: size(X_train,1)
    % node_classifiers is all the classifiers of the instance. Size: n x 1
    node_classifiers = {};
    for i=1:n_jamiul_nets
        
        % x_node: the current training instance
        x_node = X_train(node, :);
        y_train_node = y_train(node, :);        
        
        % nb_loc: neighbor locations
        nb_loc = find(neighbor_infor{1, i}(:, node));
        
        % Neighbour nodes
        X_train_cur = X_train(nb_loc, :);
        y_train_cur = y_train(nb_loc, 1);
        % X_test = [x_node_p; x_node_n];
        X_test_cur = x_node;
        y_test_cur = y_train_node;

        % [y_predicted,~]=f_multi_lin_regress_predict(X_train, ...
        %     y_train, X_test);
        m = f_train_model(X_train_cur, y_train_cur, model_type, ...
            isWekaJava, useMATLAB);
        y_predicted = f_predict_simple(X_test_cur, m, model_type,...
            isWekaJava, useMATLAB);
        
        % prediction err cannot be larger than 0.3        
        err = 0;
        if y_test_cur == 1
            err = y_test_cur - y_predicted(1);
        elseif y_test_cur == 0
            % err = y_predicted(1) -y_train_cur;
            err = y_predicted(1) ;
        end
        
        if err < err_lim
            % if passed the cutoffs, then the model is good.
            % retrain the model with the node.
            X_train2 = [x_node; X_train_cur];
            y_train2 = [y_train(node); y_train_cur];
            % b = regress(y_train2, [X_train2, ones(size(y_train2))]);
            m = f_train_model(X_train2, y_train2, model_type, ...
                isWekaJava, useMATLAB);
            
            node_classifiers=[node_classifiers; {m}];
        end
    end
    classifiers{1, node} = node_classifiers;
    node,
end



end

