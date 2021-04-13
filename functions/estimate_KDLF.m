function neuralRep                      = estimate_KDLF(estimator, MANIFOLD, TrZ, TeZ)

neuralRep.TrFA                          = MANIFOLD.TrFA;
neuralRep.TeFA                          = MANIFOLD.TeFA;
neuralRep.TrSSC                         = TrZ;
neuralRep.TeSSC                         = TeZ;
neuralRep.TrKDLF                        = cellfun(@(z)estimator(z), TrZ, 'UniformOutput',0);
neuralRep.TeKDLF                        = cellfun(@(z)estimator(z), TeZ, 'UniformOutput',0);
    