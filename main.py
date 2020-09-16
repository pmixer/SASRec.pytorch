import os
import time
import torch
import argparse

from model import SASRec
from tqdm import tqdm
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=2, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset
num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('average sequence length: %.2f' % (cc / len(user_train)))

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
model = SASRec(usernum, itemnum, args)

model.train() # enable model training
# ce_criterion = torch.nn.CrossEntropyLoss()
bce_criterion = torch.nn.BCELoss()
adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

T = 0.0
t0 = time.time()

for epoch in range(1, args.num_epochs + 1):

    for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):

        u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
        u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
        pos_pred, neg_pred = model(u, seq, pos, neg)
        pos_labels, neg_labels = torch.ones(pos_pred.shape), torch.zeros(neg_pred.shape)

        indices = np.where(pos != 0)

        loss = bce_criterion(pos_pred[indices], pos_labels[indices])
        loss += bce_criterion(neg_pred[indices], neg_labels[indices])
        loss.backward()
        adam_optimizer.step()

        print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))

    if epoch % 20 == 0:
        model.eval()
        t1 = time.time() - t0
        T += t1
        print('Evaluating', end='')
        t_test = evaluate(model, dataset, args)
        t_valid = evaluate_valid(model, dataset, args)
        print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

        f.write(str(t_valid) + ' ' + str(t_test) + '\n')
        f.flush()
        t0 = time.time()
        model.train()


f.close()
sampler.close()
print("Done")