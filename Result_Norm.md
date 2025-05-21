# Results on Layer Norm in SASRec

## Experimental Settings

We use four datasets provided in the original code, including Beauty, Movielens-1M, Video, and Steam. The hyperparameters used in the experiments are closely aligned with with those in the original paper. Specifically, we set the embedding size to 50, the learning rate to 0.001, the batch size to 128, the layer size to 2, the dropout rate to 0.2 for Movielens-1M and 0.5 for other datasets. The maximum sequence length is set to 200 for Movielens-1M and 50 for other datasets. We run maximum 500 epochs on Steam and 1000 epochs on other datasets.

Note that in this experiment, the BCE loss is used, and the sampled-based evaluation for NDCG@10 and HR@10 is applied. The more advanced Softmax Loss (SL) has already been tested in [#47](https://github.com/pmixer/SASRec.pytorch/issues/47), where the overall (non-sampled) NDCG@10 and HR@10 are evaluated.

## Results

**Analysis.** Results below show that the original SASRec design generally performs worse than the Pre-LN and Post-LN designs. Additionally, the Pre-LN design shows larger improvements than the Post-LN design in three datasets.

**NDCG@10 Results.**

| Norm Type | Beauty | Movielens-1M | Video | Steam |
| --- | :---: | :---: | :---: | :---: |
| Original SASRec | 0.3104 | 0.5946 | 0.5308 | 0.6167 |
| Pre-LN | **0.3193** | 0.5940 | **0.5376** | **0.6284** |
| Post-LN | 0.3146 | **0.5995** | 0.5297 | 0.6201 |

**HR@10 Results.**

| Norm Type | Beauty | Movielens-1M | Video | Steam |
| --- | :---: | :---: | :---: | :---: |
| Original SASRec | 0.4669 | 0.8242 | 0.7318 | 0.8629 |
| Pre-LN | **0.4784** | 0.8252 | **0.7442** | **0.8702** |
| Post-LN | 0.4696 | **0.8270** | 0.7350 | 0.8684 |

## Modifications

Given the results above and [#47](https://github.com/pmixer/SASRec.pytorch/issues/47), we suggest to use standard LN in SASRec. We modified the original norm design (cf. [lines 79-83 in model.py](https://github.com/pmixer/SASRec.pytorch/blob/main/python/model.py#L79-L83)), providing a `norm_first` option to choose the Pre-LN (True) or Post-LN (False) design.
