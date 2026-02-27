# Repo:
This repository replicates the 2018 paper "Self-Attentive Sequential Recommendation" (aka SASRec) in PyTorch (translated from the authors' Tensorflow code). The structure is

## Directory structure
CLAUDE.md
main.py -- the entry point to training and inference
model.py -- the model definition (SASRec) as nn.Module
utils.py -- utility functions, especially evaluate, evaluate_valid, evaluate_valid_with_filter
train_rl_filter.py -- trains an RL policy to select samples from the history
eval_ndcg.py -- per-user NDCG@10 on validation and test sets with a configurable filter
explore.ipynb -- an initial exploration of filtering functions
data/ -- text files containing benchmark datasets
ml-1m_default/ -- checkpoints for a model training run on MovieLens-1M
results/ -- JSON files with results for different methods and sequence lengths

# Goal:
The goal of my work is to develop sequence filtering methods to improve sequential recommender systems. The hypothesis is that if we can intelligently select the right samples in from the history, we can improve next item prediction.

# Methods:
These methods have been tried.

## Most recent (default)
Default selection is `from_end`, i.e. the most recent `max_len` samples in the history.

## K-Means Mixed unique and recent
`kmeans_<num-clusters>_mixed_unique_and_recent`: fit K-means to the table of item embeddings, infer the cluster labels for the items in a user's history. Count the number of unique lables in the history (`num_unique`). Then take `max_len - num_unique` 

## K-Means Bounded cluster filtering
Same K-means as above, but go through the sequence and select at most M samples from each cluster.

## MMR Filtering
Use Maximal Marginal Relevance `score=λ⋅recency−(1−λ)⋅maxsimilarity to selected` to score samples. Compute the similarity as dot-product.

## Difficulty based selection
Estimate loss for each item in the history. Then remove hardest or easiest.

## RL-based selection
Using RLOO with a policy where actions are to "include" an item from the history. Indices to include in the history are sampled from the policy output.

# Results
MMR Filtering has outperformed "Most recent" by a very small margin (<0.1%) on all sequence lengths. However, we are looking for better results.

