function [KDLF,mdl] = modelKDLF(Z, X)

Z_          = [Z{:}]';
X_          = [X{:}]';

n           = size(X_,1);

mdl.b       = [ones(n,1) X_]\Z_;
mdl.fcn     = @(x)mdl.b'*[ones(1,size(x,2)) ; x];
KDLF        = cellfun(@(x)mdl.fcn(x), X, 'UniformOutput',0);