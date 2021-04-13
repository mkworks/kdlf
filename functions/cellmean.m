function [mu, er] =cellmean(x)

sum_x = [];
for k = 1 : length(x)
    if ~isempty(x{k})
        sum_x = cat(3, sum_x, x{k});
    end
end
mu = mean(sum_x,3);
er = std(sum_x,[],3);