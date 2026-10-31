function Project7_QLearning()
% =========================================================================
%  MTRE 4820 FALL 2026
%  Project #7 : Robot navigation with Q-learning
%  NAME: Jenn Price
%
%  Environment : 6 states (rooms 0-4 and the outside / goal area 5)
%  Parameters  : gamma = 0.8 , alpha = 1
%  Part 1      : reward 100 for every move INTO room 5
%  Part 2      : reward for room 1 -> room 5 changed to 200

clc; close all;

%% ---------------- 1. Learning parameters ------------------------------
gamma       = 0.8;     % discount factor
alpha       = 1;       % learning rate (alpha = 1 -> update in the project handout)
goal        = 5;       % goal state (room 5 = outside / Finish)
maxEpisodes = 1000;    % upper limit on training episodes
patience    = 100;     % stop early after this many episodes with no change in Q
seed        = 4820;    % fixed random seed -> results are reproducible
snapEps     = [1 2 3]; % "first iterations" whose Q matrix is saved for the report

%% ---------------- 2. Environment (floor plan connections) ---------------
% Each row = one door (two-way connection) taken from the floor plan.
%   0-4 , 1-3 , 1-5 , 2-3 , 3-4 , 4-5   (+ 5-5 so the goal is absorbing)
doors = [0 4;
         1 3;
         1 5;
         2 3;
         3 4;
         4 5];

%% ===================== PART 1: baseline rewards =========================
R1 = buildR(doors, goal, 100, []);            % 100 for entering room 5
[Q1, snaps1, hist1, nEp1] = qLearn(R1, gamma, alpha, goal, ...
                                   maxEpisodes, patience, seed, snapEps);
reportPart('PART 1 - Reward of 100 for reaching room 5', ...
           R1, Q1, snaps1, snapEps, nEp1, goal, 2);

%% ===================== PART 2: room1 -> room5 = 200 =====================
R2 = buildR(doors, goal, 100, [1 5 200]);     % override R(1,5) = 200
[Q2, snaps2, hist2, nEp2] = qLearn(R2, gamma, alpha, goal, ...
                                   maxEpisodes, patience, seed, snapEps);
reportPart('PART 2 - Reward from room 1 to room 5 changed to 200', ...
           R2, Q2, snaps2, snapEps, nEp2, goal, 2);

%% ===================== Effect of the reward =============================
fprintf('\n================ EFFECT OF THE REWARD ================\n');
fprintf('Change in Q (Part 2 - Part 1):\n');
printQ(Q2 - Q1);
fprintf('Best action from every room:\n');
fprintf('  Room | Part 1 best next room | Part 2 best next room\n');
for s = 0:goal-1
    fprintf('   %d   |        %-14s |        %s\n', s, ...
            mat2str(bestActions(Q1, s)), mat2str(bestActions(Q2, s)));
end

%% ===================== Figures ==========================================
plotConvergence(hist1, hist2);
plotFloorPlan(doors, Q1, Q2, goal);
end


% =========================================================================
%                              LOCAL FUNCTIONS
% =========================================================================

function R = buildR(doors, goal, goalReward, overrides)
% Builds the 6x6 reward matrix.
%   -1  : no door (illegal move)
%    0  : legal move that does not reach the goal
%   goalReward : legal move into the goal state
% overrides : rows of [from to reward] that replace the default value.
n = goal + 1;
R = -ones(n);                                   % start with "no door"
for k = 1:size(doors, 1)
    a = doors(k, 1);  b = doors(k, 2);
    R(a+1, b+1) = 0;  R(b+1, a+1) = 0;          % doors are two-way
end
R(R(:, goal+1) == 0, goal+1) = goalReward;     % moves into goal get reward
R(goal+1, goal+1) = goalReward;                 % goal -> goal (absorbing)
for k = 1:size(overrides, 1)
    R(overrides(k,1)+1, overrides(k,2)+1) = overrides(k,3);
end
end


function [Q, snaps, hist, nEp] = qLearn(R, gamma, alpha, goal, ...
                                        maxEpisodes, patience, seed, snapEps)
% Tabular Q-learning exactly as described in the project handout:
%   for each episode: random start; until goal: random legal action,
%   Q(s,a) = R(s,a) + gamma * max_a' Q(s',a')   (alpha = 1)
rng(seed);                                      % reproducible results
n      = size(R, 1);
Q      = zeros(n);                              % step 2: Q = 0
snaps  = cell(1, numel(snapEps));
hist   = zeros(maxEpisodes, 1);                 % sum(Q) after each episode
nonGoal = setdiff(0:n-1, goal);
quiet  = 0;

for ep = 1:maxEpisodes
    Qold  = Q;
    state = pickRandom(nonGoal);                 % random initial state
    while state ~= goal                          % until goal is reached
        actions   = find(R(state+1, :) >= 0) - 1;      % legal next rooms
        nextState = pickRandom(actions);               % pick one at random
        nextActs  = find(R(nextState+1, :) >= 0);      % legal moves from s'
        maxNextQ  = max(Q(nextState+1, nextActs));     % max_a' Q(s',a')
        target    = R(state+1, nextState+1) + gamma * maxNextQ;
        Q(state+1, nextState+1) = (1 - alpha) * Q(state+1, nextState+1) ...
                                + alpha * target;      % = target when alpha = 1
        state = nextState;                             % move on
    end

    hist(ep) = sum(Q(:));
    idx = find(snapEps == ep);
    if ~isempty(idx), snaps{idx} = Q; end

    % ---- early stopping: Q has stopped changing -> converged ----
    if isequal(Q, Qold), quiet = quiet + 1; else, quiet = 0; end
    if quiet >= patience && ep > max(snapEps), break; end
end
nEp  = ep;
hist = hist(1:nEp);
end


function x = pickRandom(v)
% Returns one element of vector v chosen uniformly at random.
x = v(ceil(rand * numel(v)));
end


function reportPart(titleStr, R, Q, snaps, snapEps, nEp, goal, startRoom)
fprintf('\n=====================================================\n');
fprintf('%s\n', titleStr);
fprintf('=====================================================\n');
fprintf('Reward matrix R (rows = current room, cols = next room):\n');
printQ(R);
for k = 1:numel(snapEps)
    fprintf('Q matrix after episode %d:\n', snapEps(k));
    printQ(snaps{k});
end
fprintf('FINAL Q matrix (converged after %d episodes):\n', nEp);
printQ(Q);
fprintf('Normalized final Q (%% of max):\n');
printQ(round(100 * Q / max(Q(:))));
paths = allOptimalPaths(Q, startRoom, goal);
fprintf('Optimal path(s) from room %d to room %d:\n', startRoom, goal);
for p = 1:numel(paths)
    fprintf('   %s   (%d moves)\n', ...
            strjoin(arrayfun(@num2str, paths{p}, 'UniformOutput', false), ' -> '), ...
            numel(paths{p}) - 1);
end
if numel(paths) > 1
    fprintf('   (tie: these paths have exactly the same Q value)\n');
end
end


function printQ(M)
n = size(M, 1);
fprintf('        ');
fprintf('%9d', 0:n-1);  fprintf('\n');
for i = 1:n
    fprintf('  %d  | ', i-1);
    fprintf('%9.2f', M(i, :));
    fprintf('\n');
end
fprintf('\n');
end


function a = bestActions(Q, s)
% All next rooms that give the maximum Q value from room s (handles ties).
row = Q(s+1, :);
a   = find(abs(row - max(row)) < 1e-9) - 1;
end


function paths = allOptimalPaths(Q, s, goal)
% Greedy policy: from each room pick the action with the highest Q.
% If several actions tie, every tied branch is followed.
if s == goal
    paths = {goal};  return;
end
paths = {};
for a = bestActions(Q, s)
    sub = allOptimalPaths(Q, a, goal);
    for k = 1:numel(sub)
        paths{end+1} = [s sub{k}]; %#ok<AGROW>
    end
end
end


function plotConvergence(hist1, hist2)
figure('Name', 'Q-learning convergence');
plot(1:numel(hist1), hist1, 'b-', 'LineWidth', 1.5); hold on;
plot(1:numel(hist2), hist2, 'r-', 'LineWidth', 1.5);
xlabel('Episode'); ylabel('Sum of all Q values');
title('Convergence of the Q matrix');
legend('Part 1: R(1,5) = 100', 'Part 2: R(1,5) = 200', 'Location', 'southeast');
grid on;
end


function plotFloorPlan(doors, Q1, Q2, goal)
% Simple map of the rooms with the optimal path from room 2 drawn on top.
xy = [0 1; 1 1; 2 0.5; 1 0; 0 0; 1.5 1.8];      % (x,y) of rooms 0..5
Qs = {Q1, Q2};
names = {'Part 1 (R_{1\rightarrow5} = 100)', 'Part 2 (R_{1\rightarrow5} = 200)'};
figure('Name', 'Optimal path from room 2 to room 5');
for k = 1:2
    subplot(1, 2, k); hold on; axis equal off;
    for d = 1:size(doors, 1)                          % doors in grey
        p = xy(doors(d, :) + 1, :);
        plot(p(:,1), p(:,2), '-', 'Color', [0.75 0.75 0.75], 'LineWidth', 2);
    end
    paths = allOptimalPaths(Qs{k}, 2, goal);
    styles = {'r-', 'b--'};
    for p = 1:numel(paths)                             % optimal path(s)
        pts = xy(paths{p} + 1, :);
        plot(pts(:,1), pts(:,2), styles{min(p,2)}, 'LineWidth', 3);
    end
    for r = 0:goal                                     % room markers
        plot(xy(r+1,1), xy(r+1,2), 'ko', 'MarkerSize', 22, ...
             'MarkerFaceColor', [1 1 1]);
        text(xy(r+1,1), xy(r+1,2), num2str(r), 'HorizontalAlignment', 'center', ...
             'FontWeight', 'bold', 'FontSize', 12);
    end
    title(names{k});
end
end
