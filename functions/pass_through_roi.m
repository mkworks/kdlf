function y = pass_through_roi(x, roi)
try
y = x(:, roi);
catch
    y = [];
end