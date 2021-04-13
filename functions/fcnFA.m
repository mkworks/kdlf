function FA = fcnFA(cY, nLatents)

Y                               = [cY{:}];

[estParams, LL] = fastfa(Y, nLatents, 'typ', 'fa','cvf',0);
% To save disk space, don't save posterior covariance, which is identical
% for each data point and can be computed from the learned parameters.
FA.estParams    = estParams;
FA.LL           = LL;
FA.fcn          = @(y)fastfa_estep(y, FA.estParams).mean;
FA.getLL        = @(y)fastfa_estep(y, FA.estParams);

L                           = FA.estParams.L;
psi                         = FA.estParams.Ph;

[u, lambda]                 = eig(L*L','vector');
shared_var_explained_mode   = lambda / sum(lambda);
