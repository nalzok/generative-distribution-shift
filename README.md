## TODO

+ Perform grid search on MNIST to figure out the optimal hyperparameters
    + Currently the best performance is achieved by `gmm_dim48_aelr0.001_ufactor100_K6_R16_lr1_lambda0.25_kappa0.75.txt` with 94.35% validation accuracy
+ Simulate distribution shift by clipping and/or rotate the digits
+ Try TTA and see if that helps
