modified based on [paper author's tensorflow implementation](https://github.com/kang205/SASRec), switching to PyTorch(v1.6) for simplicity, executable by:

```python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2```

pls check paper author's [repo](https://github.com/kang205/SASRec) for detailed intro and more complete README, and here's paper info FYI :)

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
