function [nX, recon]        = normalize_data(X)

mu_x                        = mean(X,2);
std_x                       = std(X,[],2);
nX                          = (X - mu_x) ./ std_x;
recon1                      = @(yhat)yhat .* std_x + mu_x;

min_x                       = min(nX(:));
max_x                       = max(nX(:));
alpha                       = 1;
nX                          = (nX - min_x) ./ (max_x - min_x) * alpha;
recon2                      = @(yhat)yhat ./ alpha * (max_x - min_x) + min_x;
recon                       = @(yhat)recon1(recon2(yhat));

% max(max(abs(recon(nX) - X)))


