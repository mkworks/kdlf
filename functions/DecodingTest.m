function [CC, RMSE, Hat]    = DecodingTest(neuralRep, TrX, TeX, trtask_info, tetask_info)

dim                         = (size(TeX{1},1)-1)/3;
lists                       = fieldnames(neuralRep);
for k = 1 : length(lists)
    if strcmp(lists{k}(1:2), 'Tr')
        task_info           = trtask_info;
    else
        task_info           = tetask_info;
    end
    neuralRep.(lists{k})    = cellfun(@(x,t)x(:,t.target_cue:t.target_aqu),neuralRep.(lists{k}), task_info, 'UniformOutput',0);
end

TrX                         = cellfun(@(x,t)x(dim+1:dim*2,t.target_cue:t.target_aqu),TrX,trtask_info, 'UniformOutput',0);
TeX                         = cellfun(@(x,t)x(dim+1:dim*2,t.target_cue:t.target_aqu),TeX,tetask_info, 'UniformOutput',0);

mask                        = regexp(lists,'Tr');
mask                        = ~cellfun(@isempty,mask);
TrMask                      = lists(mask);
TeMask                      = lists(~mask);

% Decoding Progress with Linear Decoder
warning off;
num_methods                 = length(TrMask);
for k = 1 : num_methods
    LF.(TrMask{k})          = LFTraining(neuralRep.(TrMask{k}), TrX);
    Hat.(TeMask{k})         = cellfun(@(z)LFpredict(z, LF.(TrMask{k})), neuralRep.(TeMask{k}), 'UniformOutput',0);
    Hat.TeX                 = TeX;
end

% Measure decoding performance (RMSE and Correlation Coefficients)
for k = 1 : num_methods
    [RMSE_.(TeMask{k}(3:end)), CC_.(TeMask{k}(3:end))] = cellfun(@(x1, x2)evaluate_decoding(x1, x2, dim), Hat.(TeMask{k}), Hat.TeX, 'UniformOutput',0);
end


st_label                            = {'SSC','FA','KDLF'};
num_methods                         = length(st_label);
for k = 1 : num_methods
    RMSE.(st_label{k})              = cell2mat(cellfun(@(x)[x.P x.V x.A]', RMSE_.(st_label{k}), 'UniformOutput',0));
    CC.P.(st_label{k})              = cell2mat(cellfun(@(x)x.P, CC_.(st_label{k}), 'UniformOutput',0));
    CC.V.(st_label{k})              = cell2mat(cellfun(@(x)x.V, CC_.(st_label{k}), 'UniformOutput',0));
    CC.A.(st_label{k})              = cell2mat(cellfun(@(x)x.A, CC_.(st_label{k}), 'UniformOutput',0));
    CC.S.(st_label{k})              = cell2mat(cellfun(@(x)x.S, CC_.(st_label{k}), 'UniformOutput',0));
end
