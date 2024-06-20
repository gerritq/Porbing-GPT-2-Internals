# Probing Attention Heads and MLPs in GPT-2 for Truthfulness

Inspired by research on probing model internals to reveal an internal ‘belief’ about truthfulness (e.g., Li et al. 2023), I probe attention heads and MLPs in GPT-2-xl (1.5 billion parameters) with the TruthfulQA dataset (Lin et al., 2022).

I concatenate prompts with correct and incorrect answers, respectively, feed them to the model, and record internal activations of all attention heads and MLPs. I train a linear classifier (logistic regression with default parameters) with activations of the last token of an input sequence to predict the label (correct/incorrect statement) of the last token of the input sequence. I then predict the labels of the validation set for each module, which yields insights into whether there are model internals that are indicative of the truthfulness of a claim.

For GPT-2-xl with 48 layers, I find that attention heads between layers 5-18 show a prediction accuracy of truthfulness of up to 74%. For MLPs, I find that prediction accuracy peaks between layers 10-20 but then - surprisingly - monotonically declines for the remaining layers.

A great resource for collectiong model internals is this [post](https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch).
