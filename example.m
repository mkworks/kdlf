clear all; close all; clc;
addpath(genpath('./functions'));

load example_data.mat;         % load example neural data
                               % Kinematic parameters (four kinematic parameters for 2D working spaces): Position, Velocity, Acceleration, and Speed
                               % number of neurons: 101
                               % training set: 132 trials 
                               % testing set: 43 trials
                               % 20 ms bins         

binwidth_sec                                = 0.02; % 20 ms bins
n_factors                                   = 23;   % number of latent factors (empirically set up)

% Define KDLF
[KDLF, MANIFOLD]                            = makeKDLF(trainZ, testZ, trainX, testX, n_factors);

% Train KDLF estimator
KDLF_estimator                              = train_KDLF_estimator(trainZ, KDLF.TrFA, optimal_hyperparams);

% Estimate KDLF  
neuralRep                                   = estimate_KDLF(KDLF_estimator, MANIFOLD, trainZ, testZ);


% Draw KDLF temoral changes
baseline                                    = 0.2/binwidth_sec; % 200 ms befor motion onset
epocline                                    = 0.5/binwidth_sec; % 500 ms after motion onset
time                                        = (-baseline : epocline) * binwidth_sec;
speed   = cellfun(@(x, mo_i)pass_through_roi(x(end,:), mo_i - baseline : mo_i + epocline)', testX, test_motion_onset, 'UniformOutput', 0);
fa_epoc = cellfun(@(x, mo_i)pass_through_roi(x, mo_i - baseline : mo_i + epocline)', neuralRep.TeFA, test_motion_onset, 'UniformOutput', 0);
kd_epoc = cellfun(@(x, mo_i)pass_through_roi(x, mo_i - baseline : mo_i + epocline)', neuralRep.TeKDLF, test_motion_onset, 'UniformOutput', 0);

[mu_fa, er_fa]= cellmean(fa_epoc);
[mu_kd, er_kd]= cellmean(kd_epoc);

fig = figure; clf; 
subplot(311); errorbar(time, mean([speed{:}],2), std([speed{:}],[],2),'linewidth',2,'color','k','linewidth',1.5); hold on; plot([0 0],get(gca,'ylim'),'k--','linewidth',1.2); xlim([-0.2 0.5]);
subplot(312); errorbar(time, mean(mu_fa,2), mean(er_fa,2), 'color','k','linewidth',1.5); hold on; plot([0 0],get(gca,'ylim'),'k--','linewidth',1.2);xlim([-0.2 0.5]);
subplot(313); errorbar(time, mean(mu_kd,2), mean(er_kd,2), 'color','k','linewidth',1.5); hold on; plot([0 0],get(gca,'ylim'),'k--','linewidth',1.2);xlim([-0.2 0.5]); xlabel('time (sec)','fontsize',12);

% Decoding Test
[Corr, RMSE, Trajectory]                    = DecodingTest(neuralRep, trainX, testX, train_task, test_task);

CC = [Corr.S.SSC;Corr.S.FA; Corr.S.KDLF];

figure, bar(mean(CC,2),'w'); hold on; errorbar(1:3, mean(CC,2), std(CC,[],2),'LineStyle','none','color','k'); ylabel('corr'); set(gca,'xticklabel',{'SSC','FA','KDLF'});


SSC_V   = cellfun(@(x)cumsum(x')', Trajectory.TeSSC, 'UniformOutput', 0);
FA_V    = cellfun(@(x)cumsum(x')', Trajectory.TeFA, 'UniformOutput', 0);
KDLF_V  = cellfun(@(x)cumsum(x')', Trajectory.TeKDLF, 'UniformOutput', 0);
TeX     = cellfun(@(x)cumsum(x')', Trajectory.TeX, 'UniformOutput', 0);

SSC_V   = [SSC_V{:}];
FA_V    = [FA_V{:}];
KDLF_V  = [KDLF_V{:}];
TeX     = [TeX{:}];

figure, 
subplot(1,3,1);plot(TeX(1,:), TeX(2,:),'k'); hold on; plot(SSC_V(1,:), SSC_V(2,:),'r'); title('SSC'); legend('actual','predicted');
subplot(1,3,2);plot(TeX(1,:), TeX(2,:),'k'); hold on; plot(FA_V(1,:), FA_V(2,:),'r');   title('FA');
subplot(1,3,3);plot(TeX(1,:), TeX(2,:),'k'); hold on; plot(KDLF_V(1,:), KDLF_V(2,:),'r');title('KDLF');
