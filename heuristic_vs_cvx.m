% Heuristic Greedy Algorithm vs. CVX Solver
%
% This script compares the heuristic greedy algorithm with an exact convex optimization
% baseline solved by CVX in Supplementary Material, Section D.
%
% Networks:
%   ER random networks, WS small-world networks, and BA scale-free networks
%   are generated with comparable dense connectivity. The intended average
%   degree is approximately 0.5N.
%
% Target:
%   For each generated network, the target Fiedler value is set as
%
%       target_l2 = 2 * lambda_2(L_orig),
%
%  corresponding to an 50% reduction of the diffusion time scale.
%
% Heuristic method:
%   At each iteration, the current Fiedler vector is recomputed. Existing
%   edges are ranked by the first-order sensitivity
%
%       I_ij = (v_2(i) - v_2(j))^2.
%
%   The algorithm then adds weights to high-sensitivity edges using a
%   three-stage adaptive schedule.
%
% CVX baseline:
%   The CVX problem solves the SDP relaxation exactly for the same target
%   Fiedler value. The optimality accuracy is reported as
%
%       accuracy = cost_CVX / cost_heuristic * 100%.
%
% Notes:
%   - This script requires CVX and SDPT3.
%   - The heuristic part is the same greedy routine used in the Zachary
%     Karate Club example, applied here to random network ensembles.
% =========================================================================
% =========================================================================
clear; clc; close all;

% --- 1. Experimental configuration ---
N_list = [30, 50, 70, 100, 120, 150, 200]; % Network sizes
topo_names = {'ER', 'WS', 'BA'};
num_topos = length(topo_names);

results = struct();
results.N = N_list;
results.time_heu = zeros(length(N_list), num_topos);
results.time_cvx = zeros(length(N_list), num_topos);
results.accuracy = zeros(length(N_list), num_topos); 
results.iter     = zeros(length(N_list), num_topos);

fprintf('=========================begin=================================\n');

for i = 1:length(N_list)
    N = N_list(i);
    
    for t = 1:num_topos
        
        % --- A. Generate network with comparable dense connectivity ---
        while true
            if t == 1
                % 1. ER random network with expected average degree approximately 0.5N.
                p = 0.5;
                adj_orig = triu(rand(N,N) < p, 1); 
                adj_orig = adj_orig + adj_orig';
            elseif t == 2
                % 2. WS small-world network with expected average degree approximately 0.5N.
                K_ws = max(4, 2 * round(0.25 * N)); % Keep K even; average degree is approximately K
                adj_orig = generate_WS(N, K_ws, 0.2);
            elseif t == 3
                % 3. BA scale-free network with expected average degree approximately 0.5N.
                m_ba = max(2, round(0.25 * N)); 
                adj_orig = generate_BA(N, m_ba);
            end
            
            L_orig = diag(sum(adj_orig)) - adj_orig;
            lam = sort(eig(L_orig));
            if lam(2) > 1e-4, break; end % Ensure that the initial network is connected
        end
        
        l2_init = lam(2);
        target_l2 = 2 * l2_init; % Target: double the original Fiedler value.
        [rows, cols] = find(triu(adj_orig));
        num_edges = length(rows);
        
        % --- B. Run heuristic greedy algorithm
        t_start = tic;
        max_w = 2.0;                      
        min_lift_abs = l2_init * 0.001;
        
        L_curr = L_orig;
        w_heu = sparse(N,N);
        hist_l2 = l2_init;
        iter = 0;
        
        while hist_l2(end) < target_l2 && iter < 3000
            iter = iter + 1;
            [V, D_eig] = eig(full(L_curr)); 
            [lam_vals, idx] = sort(diag(D_eig));
            l2_curr = lam_vals(2); V = V(:, idx);
            progress = l2_curr / target_l2;
            
            % Three-stage adaptive schedule
            if progress < 0.80
                current_batch = max(5, round(N / 5)); step_ratio = 0.20; filter_ratio = 0.1;
            elseif progress < 0.96
                current_batch = max(2, round(N / 20)); step_ratio = 0.05; filter_ratio = 0.5;
            else
                current_batch = 1; step_ratio = 0.01; filter_ratio = 1.0;                  
            end
            
            % Edge sensitivity calculation based on Fiedler-vector gradients
            v2 = V(:,2);
            if (lam_vals(3)-lam_vals(2)) < 1e-3
                 sens = (v2(rows)-v2(cols)).^2 + (V(rows,3)-V(cols,3)).^2;
            else
                 sens = (v2(rows)-v2(cols)).^2;
            end
            
            [best_vals, best_idx] = maxk(sens, current_batch);
            threshold = best_vals(1) * filter_ratio; 
            valid = best_vals >= threshold;
            act_idx = best_idx(valid); act_sens = best_vals(valid);
            
            gap = target_l2 - l2_curr;
            lift = max(gap * step_ratio, min_lift_abs); 
            if lift > gap, lift = gap; end 
            
            if ~isempty(act_idx)
                lift_per = lift / length(act_idx);
                for k=1:length(act_idx)
                    ii = act_idx(k); S = act_sens(k);
                    dw = min(lift_per/(S+1e-15), max_w);
                    u=rows(ii); v=cols(ii);
                    L_curr(u,u)=L_curr(u,u)+dw; L_curr(v,v)=L_curr(v,v)+dw;
                    L_curr(u,v)=L_curr(u,v)-dw; L_curr(v,u)=L_curr(v,u)-dw;
                    w_heu(u,v)=w_heu(u,v)+dw; w_heu(v,u)=w_heu(v,u)+dw;
                end
            end
            [~, D_new] = eig(full(L_curr));
            lam_new = sort(diag(D_new));
            hist_l2(end+1) = lam_new(2);
        end
        t_heu = toc(t_start);
        cost_heu = full(sum(sum(triu(w_heu)))); 
        
        results.time_heu(i, t) = t_heu;
        results.iter(i, t)     = iter;
        
        % --- C. Run CVX baseline as ground truth ---
        t_cvx = NaN; cost_cvx = NaN; accuracy_percent = NaN;
        % CVX is attempted for all N. If the solver fails, NaN is recorded.
        B = sparse(N, num_edges);
        for k=1:num_edges, B(rows(k),k)=1; B(cols(k),k)=-1; end
        t_start_cvx = tic;
        try
            cvx_begin sdp quiet
                cvx_solver sdpt3 
                variable w_cvx(num_edges) nonnegative
                minimize( sum(w_cvx) )
                subject to
                    L_added = B * diag(w_cvx) * B';
                    L_orig + L_added + (target_l2/N)*ones(N) >= target_l2 * eye(N);
            cvx_end
            t_cvx = toc(t_start_cvx);
            if strcmpi(cvx_status, 'Solved') || strcmpi(cvx_status, 'Inaccurate/Solved')
                cost_cvx = cvx_optval;
                accuracy_percent = (cost_cvx / cost_heu) * 100; % Optimality accuracy
            end
        catch
            % If CVX fails, for example due to memory limits, keep NaN.
        end
        
        results.time_cvx(i, t) = t_cvx;
        results.accuracy(i, t) = accuracy_percent;
        
        if isnan(accuracy_percent), acc_str = 'NaN'; else, acc_str = sprintf('%6.2f%%', accuracy_percent); end
        fprintf('Acc: %s | Time(Heu): %5.2fs | Iter: %d\n', acc_str, t_heu, iter);
    end
end



% --- 2. Visualization ---
figure('Color','w', 'Position', [100, 100, 1000, 450]); 
colors = {[0.2, 0.6, 0.8], [0.3, 0.7, 0.5], [0.8, 0.4, 0.3]}; % Blue: ER, green: WS, red: BA
markers = {'o', 's', 'd'};

% Subplot 1: Runtime comparison
subplot(1, 2, 1); hold on; 
for t = 1:num_topos
    % Heuristic method
    plot(N_list, results.time_heu(:, t), ['-', markers{t}], 'Color', colors{t}, ...
        'LineWidth', 2, 'MarkerFaceColor', colors{t}, 'MarkerSize', 6, 'DisplayName', ['Heu: ', topo_names{t}]);
    
    valid_cvx = ~isnan(results.time_cvx(:, t));
    if any(valid_cvx)
        % CVX solver
        plot(N_list(valid_cvx), results.time_cvx(valid_cvx, t), ['--', markers{t}], 'Color', colors{t}, ...
            'LineWidth', 1.5, 'DisplayName', ['CVX: ', topo_names{t}]);
    end
end

set(gca, 'YScale', 'linear'); 

grid on; 
xticks(N_list); 
xlabel('Network Size (N)', 'FontWeight', 'bold'); 
ylabel('Runtime (seconds)', 'FontWeight', 'bold'); 
legend('Location', 'NorthWest', 'FontSize', 9); 
title('Computational Scalability (Dense)', 'FontSize', 14);

% Subplot 2: Optimality accuracy
% Accuracy is measured by the ratio of the CVX optimal total added weight
% to the heuristic total added weight:
% accuracy = cost_CVX / cost_heuristic * 100%.

subplot(1, 2, 2); 
valid_rows = ~any(isnan(results.accuracy), 2);


num_valid = sum(valid_rows);
x_indices = 1:num_valid; 
valid_N = N_list(valid_rows); 

b = bar(x_indices, results.accuracy(valid_rows, :), 'grouped', 'EdgeColor', 'none');
for t = 1:num_topos
    b(t).FaceColor = colors{t};
end

xticks(x_indices);
xticklabels(string(valid_N));

grid on; xlabel('Network Size (N)', 'FontWeight', 'bold'); ylabel('Optimality Accuracy (%)', 'FontWeight', 'bold');
title('Accuracy Across Topologies', 'FontSize', 14);
legend(topo_names, 'Location', 'SouthWest');
ylim([0, 100]); % Display accuracy from 0 to 100%.


% ========================================================
% Helper function 1: Generate a Watts-Strogatz small-world network
% ========================================================
function A = generate_WS(N, K, beta)
    A = zeros(N, N);
    for i = 1:N
        for j = 1:(K/2)
            target = mod(i+j-1, N) + 1;
            A(i, target) = 1; A(target, i) = 1;
        end
    end
    for i = 1:N
        for j = 1:(K/2)
            target = mod(i+j-1, N) + 1;
            if rand() < beta
                A(i, target) = 0; A(target, i) = 0; 
                new_target = randi(N);
                while new_target == i || A(i, new_target) == 1
                    new_target = randi(N);
                end
                A(i, new_target) = 1; A(new_target, i) = 1; 
            end
        end
    end
end

% ========================================================
% Helper function 2: Generate a Barabasi-Albert scale-free network
% ========================================================
function A = generate_BA(N, m)
    A = zeros(N, N);
    A(1:m+1, 1:m+1) = 1 - eye(m+1);
    degrees = sum(A, 2);
    for i = (m+2):N
        prob = degrees(1:i-1) / sum(degrees(1:i-1));
        chosen = zeros(1, m);
        pool_prob = prob;
        for k = 1:m
            pool_prob = pool_prob / sum(pool_prob);
            c = find(rand <= cumsum(pool_prob), 1);
            chosen(k) = c;
            pool_prob(c) = 0; 
        end
        A(i, chosen) = 1; A(chosen, i) = 1;
        degrees(i) = m;
        degrees(chosen) = degrees(chosen) + 1;
    end
end
