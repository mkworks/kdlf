function [hat,param]    = LFpredict(Z, param)

hat             = param.recon(param.beta'*[ones(1,size(Z,2)) ; (Z - param.mu')]);
