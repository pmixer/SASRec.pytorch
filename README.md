update on 05/23/2025: thx to [Wentworth1028](https://github.com/Wentworth1028) and [Tiny-Snow](https://github.com/Tiny-Snow), we have LayerNorm update, for higher NDCG&HR, and here's the [doc](https://github.com/Tiny-Snow/SASRec.pytorch/blob/main/Result_Norm.md)👍.

---

modified based on [paper author's tensorflow implementation](https://github.com/kang205/SASRec), switching to PyTorch(v1.6) for simplicity, fixed issues like positional embedding usage etc. (making it harder to overfit, except for that, in recsys, personalization=overfitting sometimes)

code in `python` folder.

to train:

```
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda
```

just inference:

```
python main.py --device=cuda --dataset=ml-1m --train_dir=default --state_dict_path=[YOUR_CKPT_PATH] --inference_only=true --maxlen=200

```

output for each run would be slightly random, as negative samples are randomly sampled, here's my output for two consecutive runs:

```
1st run - test (NDCG@10: 0.5897, HR@10: 0.8190)
2nd run - test (NDCG@10: 0.5918, HR@10: 0.8225)
```

pls check paper author's [repo](https://github.com/kang205/SASRec) for detailed intro and more complete README, and here's the paper bib FYI :)

```
@inproceedings{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}
```

I see a dozen of citations of the repo🫰, pls use the example bib as below if needed.
```
@online{huang2020sasrec_pytorch,
  author  = {Zan Huang},
  title   = {SASRec.pytorch},
  year    = {2020},
  url     = {https://github.com/pmixer/SASRec.pytorch}
}
```
