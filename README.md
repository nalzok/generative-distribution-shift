## Notice

This idea doesn't seem to work: its fails to beat Tent on CIFAT-10-C.
I'll probably revisit it later, but for now, we've moved on to another idea.

TODO:
    + Fine-tune the feature extractor in an end-to-end fashion with a GMM classification head, so that the embeddings do have a Gaussian mixture distribution.
    + Do the experiments on WILDS, which has more "typical" distribution shifts.
    + Try a larger/finer grid search for the hyperparameters.
