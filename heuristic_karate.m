% Heuristic Greedy Algorithm for Target Diffusion Time-Scale
%
%
% Example:
%   Zachary's Karate Club network with 34 nodes in the Letter.
%
% Goal:
%   Reduce the diffusion time scale by increasing the Fiedler value
%   lambda_2 of the weighted graph Laplacian. Since the diffusion time
%   scale is tau = 1/lambda_2, reducing tau by 50% is equivalent to
%   doubling lambda_2.
%
% Method:
%   At each iteration, the current Fiedler vector v_2 is recomputed.
%   Existing edges are ranked by the first-order sensitivity
%
%       I_ij = (v_2(i) - v_2(j))^2,
%
%   which approximates the increase in lambda_2 caused by adding weight
%   to edge (i,j):
%
%       Delta lambda_2 approx Delta w_ij * I_ij.
%
%   The algorithm then selects high-sensitivity edges and adds weights
%   so as to close a fixed fraction of the remaining gap to the target
%   Fiedler value.
%
% Three-stage schedule:
%   progress < 0.80       : large steps, multiple edges
%   0.80 <= progress <0.96: smaller steps, fewer edges
%   progress >= 0.96      : fine-tuning with one edge
%
% Numerical safeguards:
%   max_w        limits the maximum single-step weight increment.
%   min_lift_abs prevents asymptotic stalling near the target.
%   If lambda_2 and lambda_3 are nearly degenerate, the third eigenmode
%   is included in the sensitivity score to stabilize edge selection.
%
% Output:
%   w_heu_full is the added-weight matrix produced by the heuristic.
%
% Note:
%   This script demonstrates the practical all-edge heuristic.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

fprintf('==========================================================\n');
fprintf('Start Running: Heuristic Greedy Algorithm\n');
fprintf('==========================================================\n');

% 1. Input Data: Zachary's Karate Club Network
% A is the adjacency matrix of the original network.

A=[0,1,1,1,1,1,1,1,1,0,1,1,1,1,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0;1,0,1,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0;1,1,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0;1,1,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1;0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1;1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1;0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1;1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1;1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,1,1;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1;0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1;0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,1;0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1;1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,1;0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,1,0,1,0,1,1,0,0,0,0,0,1,1,1,0,1;0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,1,1,0,0,1,1,1,1,1,1,1,0];

[n, ~] = size(A);
deta = sum(A, 2);
L_orig = diag(deta) - A;

% Compute the initial Fiedler value.
[V_orig, D_orig] = eig(L_orig);
lam_orig = sort(diag(D_orig));
l2_init = lam_orig(2);

% Target Fiedler value.
% In the manuscript example, the target reduces the diffusion time scale
% by 50%, which corresponds to increasing lambda_2 to approximately twice its original value. 
target_l2 = 0.94;

% Extract all existing edges in the network. The heuristic allocates additional weights only to these candidate edges.
[rows, cols] = find(triu(A));
num_edges = length(rows);

fprintf('Network size N = %d, Edges M = %d\n', n, num_edges);
fprintf('Initial Fiedler value (lambda_2) = %.4f\n', l2_init);
fprintf('Target Fiedler value             = %.4f\n', target_l2);
fprintf('----------------------------------------------------------\n');

% 2. Algorithm Initialization
t_start = tic;

% Numerical safeguards.
max_w = 2.0;                    % Maximum allowed weight increment per step.
min_lift_abs = l2_init * 0.001; % Minimum expected lift per step.

L_curr = L_orig;
w_heu = sparse(n, n);           % Added-weight matrix.
hist_l2 = l2_init;
iter = 0;

% 3. Core Iteration
while hist_l2(end) < target_l2 && iter < 3000
    iter = iter + 1;

    % Compute eigenvalues and eigenvectors of the current weighted Laplacian.
    [V, D_eig] = eig(full(L_curr));
    [lam_vals, idx] = sort(diag(D_eig));
    l2_curr = lam_vals(2);
    V = V(:, idx);

    % Current progress toward the target Fiedler value.
    progress = l2_curr / target_l2;

    % ============================================================
    % Three-stage adaptive schedule.
    % ============================================================
    if progress < 0.80
        % Stage 1: large steps and multiple selected edges.
        current_batch = max(5, round(n / 5));
        step_ratio    = 0.20;
        filter_ratio  = 0.1;
    elseif progress < 0.96
        % Stage 2: medium steps and narrower edge selection.
        current_batch = max(2, round(n / 20));
        step_ratio    = 0.05;
        filter_ratio  = 0.5;
    else
        % Stage 3: fine-tuning near the target.
        current_batch = 1;
        step_ratio    = 0.01;
        filter_ratio  = 1.0;
    end

    % Edge sensitivity based on the current Fiedler vector.
    v2 = V(:,2);
    if (lam_vals(3) - lam_vals(2)) < 1e-3
        % Near-degenerate case: include the next eigenmode for stability.
        sens = (v2(rows)-v2(cols)).^2 + (V(rows,3)-V(cols,3)).^2;
    else
        sens = (v2(rows)-v2(cols)).^2;
    end

    % Select high-sensitivity edges.
    [best_vals, best_idx] = maxk(sens, current_batch);
    threshold = best_vals(1) * filter_ratio;
    valid = best_vals >= threshold;
    act_idx = best_idx(valid);
    act_sens = best_vals(valid);

    % Expected increase in lambda_2 for this iteration.
    gap = target_l2 - l2_curr;
    lift = max(gap * step_ratio, min_lift_abs);
    if lift > gap
        lift = gap;
    end

    % Add weights to selected edges using first-order perturbation:
    % Delta w_ij approx Delta lambda_2 / I_ij.
    if ~isempty(act_idx)
        lift_per = lift / length(act_idx);
        for k = 1:length(act_idx)
            ii = act_idx(k);
            S = act_sens(k);

            dw = min(lift_per / (S + 1e-15), max_w);

            u = rows(ii);
            v = cols(ii);

            % Update the weighted Laplacian.
            L_curr(u,u) = L_curr(u,u) + dw;
            L_curr(v,v) = L_curr(v,v) + dw;
            L_curr(u,v) = L_curr(u,v) - dw;
            L_curr(v,u) = L_curr(v,u) - dw;

            % Record the added edge weight.
            w_heu(u,v) = w_heu(u,v) + dw;
            w_heu(v,u) = w_heu(v,u) + dw;
        end
    end

    [~, D_new] = eig(full(L_curr));
    lam_new = sort(diag(D_new));
    hist_l2(end+1) = lam_new(2);
end

t_heu = toc(t_start);

% Total added weight.
cost_heu = full(sum(sum(triu(w_heu))));

% 4. Print Results
fprintf('==========================================================\n');
fprintf('Heuristic Completed!\n');
fprintf('==========================================================\n');
fprintf('Achieved algebraic connectivity  : %.4f\n', hist_l2(end));
fprintf('Total iterations           : %d\n', iter);
fprintf('Computation time           : %.4f seconds\n', t_heu);
fprintf('[Final Result] Total weight cost : %.4f\n', cost_heu);

w_heu_full = full(w_heu);

fprintf('The added weight matrix is saved as ''w_heu_full''.\n');

% Output the edges with the largest allocated weights.
fprintf('\nTop edges with the largest allocated weights:\n');

[all_u, all_v] = find(triu(A));
num_all_edges = length(all_u);

edge_weights = zeros(num_all_edges, 1);
for k = 1:num_all_edges
    edge_weights(k) = w_heu(all_u(k), all_v(k));
end

[sorted_weights, sort_idx] = sort(edge_weights, 'descend');

for k = 1:min(10, num_all_edges)
    if sorted_weights(k) > 1e-4
        real_u = all_u(sort_idx(k));
        real_v = all_v(sort_idx(k));
        fprintf('Edge %2d - %2d : Weight = %.4f\n', ...
            real_u, real_v, sorted_weights(k));
    end
end