function [net,recon]        = train_lstm(XTrain, YTrain, optVars)

[YTrain, recon]             = normalize_data(YTrain);

init_learning_rate          = optVars.initLearningRate;
minibatch                   = optVars.minibatch;
numHiddenUnits1             = optVars.numHiddenNodes1;
numHiddenUnits2             = optVars.numHiddenNodes2;
GradDecayFactor             = optVars.GradDecayFactor;
SqrtGradDecayFactor         = optVars.SqrtGradDecayFactor;
LR_dropFactor               = optVars.LR_dropFactor;
LR_dropPeriod               = optVars.LR_dropPeriod;
l2penalty                   = optVars.l2penalty;

[InputSize,N]               = size(XTrain);
OutputSize                  = size(YTrain,1);

layers                      = [ sequenceInputLayer(InputSize)
                                makeLayers(numHiddenUnits1, numHiddenUnits2)
                                fullyConnectedLayer(OutputSize) 
                                regressionLayer];
S.trid                      = 1 : floor(N * 0.8);
S.teid                      = 1 + floor(N * 0.8) : N;
TrXValidation               = XTrain(:,S.trid);
TrYValidation               = YTrain(:,S.trid);
TeXValidation               = XTrain(:,S.teid);
TeYValidation               = YTrain(:,S.teid);
validationFrequency         = floor(size(TeXValidation,2)/minibatch);
options                     = trainingOptions(  'adam',...
    'ExecutionEnvironment','gpu', ...
    'GradientDecayFactor',GradDecayFactor,...
    'SquaredGradientDecayFactor',SqrtGradDecayFactor,...
    'InitialLearnRate', init_learning_rate,...
    'LearnRateDropFactor', LR_dropFactor,...
    'LearnRateDropPeriod', LR_dropPeriod,...
    'LearnRateSchedule','piecewise',...
    'MaxEpochs',5000, ...
    'MiniBatchSize',minibatch,...
    'L2Regularization',l2penalty,...
    'Shuffle','every-epoch',...
    'Plots','none',...%'training-progress',...%'none',...%%
    'ValidationData',{TeXValidation,TeYValidation},...
    'ValidationPatience',Inf,...
    'ValidationFrequency',validationFrequency,...
    'OutputFcn',@(info)stopIfAccuracyNotImproving(info,15));


net                         = trainNetwork(TrXValidation, TrYValidation,layers,options);
close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_FIGURE'))

end

function stop = stopIfAccuracyNotImproving(info,N)
stop = false;

% Keep track of the best validation accuracy and the number of validations for which
% there has not been an improvement of the accuracy.
persistent bestValAccuracy
persistent valLag
persistent smoothed_rmse

% Clear the variables when training starts.
if info.State == "start"
    bestValAccuracy = 1000;
    valLag = 0;
    smoothed_rmse = 1000;
elseif ~isempty(info.ValidationRMSE)
    smoothed_rmse = smoothed_rmse * 0.5 + info.ValidationRMSE * 0.5;
    if smoothed_rmse < bestValAccuracy
        valLag = 0;
        bestValAccuracy = smoothed_rmse;
    else
        valLag = valLag + 1;
    end
    
    % If the validation lag is at least N, that is, the validation accuracy
    % has not improved for at least N validations, then return true and
    % stop training.
    if valLag >= N
        stop = true;
    end
    
end

end

function layers = makeLayers(varargin)
numLayers                           = length(varargin);
layers                              = [];
for l = 1 : numLayers
    layers                          = [layers;lstmLayer(varargin{l},'OutputMode','sequence');dropoutLayer(0.2)];
end
end
