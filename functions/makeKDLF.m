function [KDLF, MANIFOLD]               = makeKDLF(TrZ, TeZ, TrX, TeX, dims)
if exist(sprintf('./Parameters/KDLF_param.mat'),'file') == 0
    if exist('./Parameters','file') == 0
        mkdir('./Parameters/');
    end
    fprintf('Performing FA...');
    tic; fa_mdl                         = fcnFA(TrZ, dims);   duration.fa     = toc;
    fprintf('!\n');
    
    fprintf('Estimating manifolds from firing rates... \n');
    % Calculate Latent Variables
    MANIFOLD.TrFA                       = cellfun(@(z)fa_mdl.fcn(z), TrZ, 'UniformOutput',0);
    MANIFOLD.TeFA                       = cellfun(@(z)fa_mdl.fcn(z), TeZ, 'UniformOutput',0);
    
    fprintf('Complete to compute KDLF.\n');
    
    % Calculate KDLF
    [~,FA_KDLF_mdl]                     = modelKDLF(MANIFOLD.TrFA, TrX);
    
    KDLF.TrFA                           = cellfun(@(z)FA_KDLF_mdl.fcn(z), TrX, 'UniformOutput',0);
    KDLF.TeFA                           = cellfun(@(z)FA_KDLF_mdl.fcn(z), TeX, 'UniformOutput',0);
    fprintf('Saving data..\n');
    save(sprintf('./Parameters/KDLF_param.mat'), 'KDLF', 'MANIFOLD');
    fprintf('Completed.\n');    
else
    fprintf('Loading...');
    load(sprintf('./Parameters/KDLF_param.mat'));
    fprintf('!\n');
end