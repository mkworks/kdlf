function param  = LFTraining(Z_, X_)

Z               = [Z_{:}]';
X               = [X_{:}]';
muX             = mean(X,1);
erX             = std(X,[],1);
X               = (X - muX) ./ erX;
recon           = @(x)x .* erX' + muX';

nanid           = isnan(sum(Z,2));
Z(nanid,:)      = [];
X(nanid,:)      = [];

[N, M]          = size(Z);

mu              = mean(Z,1);
Zi              = [ones(N,1) Z - mu];
param.dim       = size(X,2);
for k = 1 : param.dim
    beta(:,k)   = regress(X(:,k), Zi);
end
param.beta      = beta;%X' * Zi * (Zi' * Zi)^-1;
param.mu        = mu;
param.tab       = 1;
param.decoder   = 'lf';
param.recon     = recon;