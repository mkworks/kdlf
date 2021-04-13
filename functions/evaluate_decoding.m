function [RMSE, CC]         = evaluate_decoding(hat, actual, dim)

RMSE.V = CalcED(hat', actual');
RMSE.P = CalcED(cumsum(hat'), cumsum(actual'));
RMSE.A = CalcED(diff(hat'), diff(actual'));

CC.V = (diag(corr(hat', actual')));
CC.P = (diag(corr(cumsum(hat'), cumsum(actual'))));
CC.A = (diag(corr(diff(hat'), diff(actual'))));
CC.S = corr(sqrt(sum(actual.^2,1))', sqrt(sum(hat.^2,1))');
