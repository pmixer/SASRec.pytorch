import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
from tqdm import tqdm

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

# 

# train/val/test data generation
def data_partition(fname, split='ratio'):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i, _, _, _ = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    if split == 'ratio':
        for user in User:
            nfeedback = len(User[user])
            if nfeedback < 3:
                user_train[user] = User[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                # 8:1:1で分割
                train_len = int(nfeedback * 0.8)
                valid_len = int(nfeedback * 0.1)
                test_len = nfeedback - train_len - valid_len
                user_train[user] = User[user][:train_len]
                user_valid[user] = User[user][train_len:train_len + valid_len]
                user_test[user] = User[user][train_len + valid_len:]

    elif split == 'LOO':
        for user in User:
            nfeedback = len(User[user])
            if nfeedback < 3:
                user_train[user] = User[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                user_train[user] = User[user][:-2]
                user_valid[user] = []
                user_valid[user].append(User[user][-2])
                user_test[user] = []
                user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# evaluate
def evaluate(model, dataset, args, mode):
    assert mode in {'valid', 'test'}, "mode must be either 'valid' or 'test'"
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    RECALL = 0.0
    MRR = 0.0
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in tqdm(users):

        if len(train[u]) < 1 or len(valid[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if mode == 'test':
            for i in valid[u]:
                seq[idx] = i
                idx -= 1
                if idx == 0: break
        for i in reversed(train[u]):
            if idx == 0: break
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        if mode == 'valid':
            item_idx = valid[u]
        elif mode == 'test':
            item_idx = test[u]
        
        correct_len = len(item_idx)

        # ランダムに選んだ100個のアイテムと正解データをモデルがどのように予測するか（全アイテムでやると時間がかかりすぎるため、一度やってみてもいいが）
        # for _ in range(100):
        #     t = np.random.randint(1, itemnum + 1)
        #     while t == 0: t = np.random.randint(1, itemnum + 1) # item_id=0は存在しない（パディング）のでやり直し
        #     item_idx.append(t)
        # itemnum個の配列を作成
        t = np.arange(1, itemnum + 1)
        # item_idxに含まれないアイテムを取得
        t = np.setdiff1d(t, item_idx) 
        item_idx.extend(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]  # - for 1st argsort DESC

        ranks = predictions.argsort().argsort()[0:correct_len].tolist() # 正解データの疑似ランクを取得

        valid_user += 1

        # 20未満のアイテム数をカウント
        c = 0
        top = 20
        h = 0
        for r in ranks:
            if r < 10: # この数字はtopkによる
                c += 1
                h = 1
                if r < top:
                    top = r
        
        RECALL += c / correct_len
        HT += h
        if top < 20:
            MRR += 1.0 / (top + 1)

        # if rank < 10:
        #     # RECALL += 
        #     MRR += 1.0 / (rank + 1)
        #     NDCG += 1 / np.log2(rank + 2)
        #     HT += 1
        # if valid_user % 100 == 0:
        #     print('.', end="")
        #     sys.stdout.flush()

    return RECALL / valid_user, MRR / valid_user, HT / valid_user
