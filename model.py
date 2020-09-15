import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate): # wried, why fusion X 2?
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.dropout1(self.conv1(inputs))))
        outputs += inputs
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        self.user_num = user_num
        self.item_num = item_num
        self.emb_dim = args.hidden_units

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        self.item_emb = torch.nn.Embedding(self.item_num, self.emb_dim, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.pos_emb = torch.nn.Embedding(args.maxlen, self.emb_dim) # TO IMPROVE
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, user_ids, item_seqs, pos_items, ): # for training
        # TODO: (more proper?) Positional Encoding
  
        seqs = self.item_emb[item_seqs]
        positions = np.tile(np.array(range(item_seqs.shape[1])), [item_seqs.shape[0], 1])
        seqs += self.pos_emb[positions] # seems wrong/useless 'positional embedding'
        seqs = self.emb_dropout(seqs)

        # TODO: mask 0th items(placeholder for dry-run) in item_seqs
        # would be easier if 0th item could be an exception for training
        # mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)
        # seqs *= mask # by broadcast

        for i in range(len(self.attention_layers)):
            # Self-attention, Q=layernorm(seqs), K=V=seqs
            seqs = torch.transpose(seqs, 0, 1) # (N, T, C) -> (T, N, C)
            Q = self.attention_layernorms[i](seqs) # pt attn requires time first format
            seqs = self.attention_layers[i](Q, seqs, seqs, key_padding_mask=mask, need_weights=False)
            seqs = torch.transpose(seqs, 0, 1) # (T, N, C) -> (N, T, C)

            # Point-wise Feed-forward, actually 2 Conv1D for channel wise fusion
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= mask

        seqs = self.last_layernorm(seqs)

        # TODO: seq, pos and neg reshape to (N*T, C) and embedding lookup

        # TODO: get test_item indices of while-item-set/101 length
        # then seqs * test_seqs.T for logits
        # then recover batch dim by reshape to (N, L, 101/whole-set-length)
        # then retrive last prediction by (N, -1, 101/whole-set-length) as test_logits

        # TODO: prediction
        # pos_emb * seq_emb then reduce_sum by channel dim, as pos_logits
        # neg_emb * seq_emb then reduce_sum by channel dim, as neg_logits

        # TODO: ignore padding items (0) and do logistic+cross entropy loss in current batch
        # ignore those logits filled with 0, why? is that for removing 0th item effect? what if vec are orthogonal?
        # istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        # self.loss = tf.reduce_sum(
        #     - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
        #     tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        # ) / tf.reduce_sum(istarget)
        # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # self.loss += sum(reg_losses)

        # self.auc = tf.reduce_sum(
        #     ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        # ) / tf.reduce_sum(istarget)

        # self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
        # self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def predict(self, user_ids, item_seqs, item_idx): # for 
        # get users, seqs, to be ranked item_indices and output test_logits
        pass