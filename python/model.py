import numpy as np
import torch



class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        
#         maxlen = args.maxlen
        maxlen = 200

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()
            
    def get_item_embs(self, log_seqs):
        return self.item_emb(torch.tensor(log_seqs, dtype=torch.long, device=self.dev))

    def log2feats(self, log_seqs, causal=True): # TODO: fp64 and int64 as default in python, trim?
        log_seqs_t = torch.tensor(log_seqs, dtype=torch.long, device=self.dev)
        seqs = self.item_emb(log_seqs_t)
        seqs *= self.item_emb.embedding_dim ** 0.5
        B, L = log_seqs.shape
        poss = torch.arange(1, L + 1, dtype=torch.long, device=self.dev).unsqueeze(0).expand(B, -1)
        poss = poss * (log_seqs_t != 0)
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # time dim len for enforce causality
        if causal:
            attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        else:
            attention_mask = None

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x,
                                                attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs,
                                                attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.tensor(pos_seqs, dtype=torch.long, device=self.dev))
        neg_embs = self.item_emb(torch.tensor(neg_seqs, dtype=torch.long, device=self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.tensor(item_indices, dtype=torch.long, device=self.dev)) # (I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)


class PolicyNetwork(torch.nn.Module):
    """Transformer-based policy network for RL-based sequence filtering.

    Produces per-item importance weights over a user's history.  Items are
    sampled without replacement during training (REINFORCE) and selected
    via deterministic top-k at inference.

    Item embeddings are copied from a pre-trained SASRec model and frozen;
    all other parameters are trained end-to-end.

    Long-sequence truncation: if len(sequence) > maxlen, only the most
    recent `maxlen` items are fed to the network.  Items outside that window
    receive -inf log-probability and can never be selected.
    """

    def __init__(self, user_num, item_num, args):
        super(PolicyNetwork, self).__init__()

        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.attention = getattr(args, 'attention', 'full')
        self.maxlen = args.maxlen  # = policy_maxlen
#         self.position_bias = torch.nn.Parameter(torch.ones(self.maxlen,) * 0.01, requires_grad=True)
        self.position_bias = torch.nn.Parameter(torch.arange(self.maxlen,) * 0.01, requires_grad=True)

        self.item_emb = torch.nn.Embedding(item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        # User embedding: zero-initialised so training starts user-agnostic,
        # then diverges as gradients push different users apart.
        # Disabled when user_num == 0.
        if user_num > 0:
            self.user_emb = torch.nn.Embedding(user_num + 1, args.hidden_units, padding_idx=0)
            torch.nn.init.zeros_(self.user_emb.weight)
        else:
            self.user_emb = None
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                         args.num_heads,
                                                         args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        self.out_proj = torch.nn.Linear(args.hidden_units, 1)

    def log2feats(self, log_seqs, user_ids=None):
        log_seqs_t = torch.tensor(log_seqs, dtype=torch.long, device=self.dev)
        seqs = self.item_emb(log_seqs_t)
        seqs *= self.item_emb.embedding_dim ** 0.5
        B, L = log_seqs.shape
        poss = torch.arange(1, L + 1, dtype=torch.long, device=self.dev).unsqueeze(0).expand(B, -1)
        poss = poss * (log_seqs_t != 0)
        seqs += self.pos_emb(poss)
        if user_ids is not None and self.user_emb is not None:
            u_emb = self.user_emb(torch.tensor(user_ids, dtype=torch.long, device=self.dev))  # [B, hidden]
            seqs = seqs + u_emb.unsqueeze(1)  # broadcast over sequence positions
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]
        if self.attention == 'left':
            attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        elif self.attention == 'right':
            attention_mask = ~torch.triu(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        else:
            attention_mask = None

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x,
                                                attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs,
                                                attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def filter(self, sequence, max_length):
        """Deterministic top-k inference filter.

        Selects the max_length items with the highest policy importance weights,
        restoring chronological order.  If len(sequence) <= max_length, returns
        sequence unchanged.
        """
        if len(sequence) <= max_length:
            return sequence
        with torch.no_grad():
            log_probs = self.get_log_probs(sequence)           # [min(L, maxlen)]
        k = min(max_length, log_probs.shape[0])
        top_indices = log_probs.topk(k).indices.sort().values
        offset = max(0, len(sequence) - self.maxlen)
        return [sequence[offset + i.item()] for i in top_indices]

    def rollout(self, sequence, max_length, num_samples):
        """Batched stochastic rollout for REINFORCE training.

        Returns (batch_seqs, sample_log_probs) or None if the policy's visible
        window (capped at self.maxlen) is shorter than max_length, making
        without-replacement sampling impossible.

        Returns:
            tuple: (batch_seqs, sample_log_probs) where
                batch_seqs:       np.ndarray [num_samples, max_length] — SASRec input
                sample_log_probs: Tensor [num_samples] — differentiable
            None: when visible window < max_length.
        """
        log_probs_all = self.get_log_probs(sequence)           # [min(L, maxlen)]
        if log_probs_all.shape[0] < max_length:
            return None

        probs = log_probs_all.detach().exp()
        all_selected = torch.multinomial(
            probs.unsqueeze(0).expand(num_samples, -1),
            max_length, replacement=False,
        )
        all_selected, _ = all_selected.sort(dim=1)             # chronological

        sample_log_probs = log_probs_all[all_selected].sum(dim=1)  # [num_samples]

        offset = max(0, len(sequence) - self.maxlen)
        seq_window_arr = np.array(sequence[offset:], dtype=np.int32)
        batch_seqs = seq_window_arr[all_selected.cpu().numpy()]    # [num_samples, max_length]

        return batch_seqs, sample_log_probs

    def get_log_probs(self, sequence, user_id=0):
        """Return log-softmax importance weights for each item in `sequence`.

        If len(sequence) > self.maxlen, only the most recent self.maxlen
        items are processed.  The returned tensor covers only those items;
        earlier items are implicitly assigned -inf and are never selected.

        Args:
            sequence: list of item indices (ints), length L.
            user_id:  user index (int); 0 means no user conditioning.

        Returns:
            Tensor of shape [min(L, self.maxlen)] with log-softmax weights.
        """
        if len(sequence) > self.maxlen:
            sequence = sequence[-self.maxlen:]

        log_seq = np.array(sequence, dtype=np.int32)[np.newaxis]       # [1, L]
        user_ids = np.array([user_id], dtype=np.int32)                  # [1]
        feats = self.log2feats(log_seq, user_ids)                       # [1, L, hidden]
        logits = self.out_proj(feats[0]).squeeze(-1)                    # [L]
#         import pdb; pdb.set_trace()
#         logits += torch.cumsum(self.position_bias[-logits.shape[0]:], dim=0)
        logits += self.position_bias[-logits.shape[0]:]
        return torch.nn.functional.log_softmax(logits, dim=0)          # [L]
