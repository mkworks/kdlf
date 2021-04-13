function LSTMpredict                    = train_KDLF_estimator(TrZ, TrKDLF, optimal_hyperparams)
if exist(sprintf('./Parameters/lstm_model.mat'),'file') == 0
    tic;
    [lstm_net, recon]                   = train_lstm([TrZ{:}], [TrKDLF{:}], optimal_hyperparams);
    LSTMpredict                         = @(z)recon(predict(lstm_net,z));
    lstm_take_time                      = toc;
    save(sprintf('./Parameters/lstm_model.mat'), 'lstm_net','LSTMpredict', 'lstm_take_time');
else
    load(sprintf('./Parameters/lstm_model.mat'));
end