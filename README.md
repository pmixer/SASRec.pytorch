update on 05/23/2025: thx to [Wentworth1028](https://github.com/Wentworth1028) and [Tiny-Snow](https://github.com/Tiny-Snow), we have LayerNorm update, for higher NDCG&HR, and here's the [doc](https://github.com/Tiny-Snow/SASRec.pytorch/blob/main/Result_Norm.md)👍.

update on 04/13/2025: in https://arxiv.org/html/2504.09596v1, I listed the ideas worth to try but not yet due to my limited bandwidth in sparse time.

pls feel free to do these experiments to have fun, and pls consider citing the article if it somehow helps in your recsys exploration:

```
@article{huang2025revisiting_sasrec,
  title={Revisiting Self-Attentive Sequential Recommendation},
  author={Huang, Zan},
  journal={CoRR},
  volume={abs/2504.09596},
  url={https://arxiv.org/abs/2504.09596},
  eprinttype={arXiv},
  eprint={2504.09596},
  year={2025}
}
```

or this bib for short

```
@article{huang2025revisiting,
  title={Revisiting Self-Attentive Sequential Recommendation},
  author={Huang, Zan},
  journal={arXiv preprint arXiv:2504.09596},
  year={2025}
}
```

paper source code in `latex` folder.

for questions or collaborations, pls create a new issue in this repo or drop me an email using the email address as shared.

---

modified based on [paper author's tensorflow implementation](https://github.com/kang205/SASRec), switching to PyTorch(v1.6) for simplicity, fixed issues like positional embedding usage etc. (making it harder to overfit, except for that, in recsys, personalization=overfitting sometimes)

code in `python` folder.

to train:

```
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda
```

just inference:

```
python main.py --device=cuda --dataset=ml-1m --train_dir=default --state_dict_path='ml-1m_default/SASRec.epoch=1000.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth' --inference_only=true --maxlen=200

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

I see a dozen of citations of the repo recently🫰, here's the repo bib if needed.
```
@software{Huang_SASRec_pytorch,
author = {Huang, Zan},
title = {PyTorch implementation for SASRec},
url = {https://github.com/pmixer/SASRec.pytorch},
year={2020}
}
```
