import os
import time
import torch
import argparse
import wandb
from tqdm import tqdm

from model import SASRec
from sasrec_repeat_emb import SASRec_RepeatEmb
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--project', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--split', default='ratio', type=str)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

wandb.init(
    project=f"{args.project}",
    name=f"{args.model}", 
    config={
        'dataset': args.dataset,
        'model': args.model,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'maxlen': args.maxlen,
        'hidden_units': args.hidden_units,
        'num_blocks': args.num_blocks,
        'num_epochs': args.num_epochs,
        'num_heads': args.num_heads,
        'dropout_rate': args.dropout_rate,
        'l2_emb': args.l2_emb,
        'device': args.device,
        'inference_only': args.inference_only,
        'state_dict_path': args.state_dict_path,
        'split': args.split
    }
    )

if __name__ == '__main__':
    # global dataset
    dataset = data_partition(args.dataset, args.split)

    [user_train, user_valid, user_test, repeat_train, repeat_valid, repreat_test, usernum, repeatnum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print(f'user num {usernum}')
    print(f'item num {itemnum}')
    print(f'max repeat {repeatnum}')
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    
    sampler = WarpSampler(user_train, repeat_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    if args.model == 'SASRec':
        model = SASRec(usernum, itemnum, args).to(args.device)
    elif args.model == 'SASRec_RepeatEmb':
        model = SASRec_RepeatEmb(usernum, itemnum, repeatnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    
    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
            
    
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, args.model, dataset, args, mode='test')
        print('test (Rcall@10: %.4f, MRR@10 %.4f, HR@10: %.4f)' % (t_test[0], t_test[1], t_test[2]))
    
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    T = 0.0
    t0 = time.time()

    early_stop = -1
    early_count = 0
    best_epoch = 0
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        print('epoch: ', epoch)
        for step in tqdm(range(num_batch)): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, repeat, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, repeat, pos, neg = np.array(u), np.array(seq), np.array(repeat), np.array(pos), np.array(neg)
            if args.model == 'SASRec':
                pos_logits, neg_logits = model(u, seq, pos, neg)
            elif args.model == 'SASRec_RepeatEmb':
                pos_logits, neg_logits = model(u, seq, repeat, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
    
        if epoch % 1 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating')
            t_valid = evaluate(model, args.model, dataset, args, mode='valid')
            
            # early stopping
            if early_stop < t_valid[3]:
                early_stop = t_valid[3] # MRR@20
                best_model_params = model.state_dict().copy()  # 最高のモデルのパラメータを一時的に保存
                best_epoch = epoch
                early_count = 0
            else:
                early_count += 1
            
            print('epoch:%d, time: %f(s), valid (Rcall@10: %.4f, Rcall@20: %.4f, MRR@10: %.4f, MRR@20: %.4f, HR@10: %.4f, HR@20: %.4f))'
                    % (epoch, T, t_valid[0], t_valid[1], t_valid[2],  t_valid[3], t_valid[4], t_valid[5]))
    
            f.write(str(t_valid) + '\n')
            f.flush()            
            t0 = time.time()
            model.train()

            wandb.log({"epoch": epoch, "time": T, "valid_Rcall@10": t_valid[0], "valid_Rcall@20": t_valid[1], "valid_MRR@10": t_valid[2], "valid_MRR@20": t_valid[3], "valid_HR@10": t_valid[4], "valid_HR@20": t_valid[5]})
        
        if early_count == 10:
            print('early stop at epoch {}'.format(epoch))
            print('testing')
            folder = args.model + '_' + args.dataset + '_' + args.train_dir
            fname = 'BestModel.MRR={}.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(early_stop, best_epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(best_model_params, os.path.join(folder, fname))

            # 最も評価指標が高かったエポックのモデルのパスを指定します。
            best_model_path = os.path.join(folder, fname)

            # モデルの重みをロードします。
            model.load_state_dict(torch.load(best_model_path))

            # ロードした重みを用いてテストの評価を行います。
            t_test = evaluate(model, args.model, dataset, args, mode='test')
            print('best epoch:%d, time: %f(s), test (Rcall@10: %.4f, Rcall@20: %.4f, MRR@10: %.4f, MRR@20: %.4f, HR@10: %.4f, HR@20: %.4f)'
                    % (best_epoch, T, t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5]))
            f.write(str(t_test) + '\n')
            f.flush()

            wandb.log({"best_epoch": best_epoch, "time": T, "test_Rcall@10": t_test[0], "test_Rcall@20": t_test[1], "test_MRR@10": t_test[2], "test_MRR@20": t_test[3], "test_HR@10": t_test[4], "test_HR@20": t_test[5]})
            
            break
    
        if epoch == args.num_epochs:
            print('testing')
            folder = args.dataset + '_' + args.train_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))

            # 最も評価指標が高かったエポックのモデルのパスを指定します。
            best_model_path = os.path.join(folder, fname)

            # モデルの重みをロードします。
            model.load_state_dict(torch.load(best_model_path))

            # ロードした重みを用いてテストの評価を行います。
            t_test = evaluate(model, args.model, dataset, args, mode='test')
            print('epoch:%d, time: %f(s), test (Rcall@10: %.4f, Rcall@20: %.4f, MRR@10: %.4f, MRR@20: %.4f, HR@10: %.4f, HR@20: %.4f)'
                    % (epoch, T, t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5]))
            f.write(str(t_test) + '\n')
            f.flush()

            wandb.log({"best_epoch": best_epoch, "time": T, "test_Rcall@10": t_test[0], "test_Rcall@20": t_test[1], "test_MRR@10": t_test[2], "test_MRR@20": t_test[3], "test_HR@10": t_test[4], "test_HR@20": t_test[5]})
    
    f.close()
    sampler.close()
    wandb.finish()
    print("Done")
