use candle_core::{Device, Module, Result, Tensor, DType, D};
use candle_nn::{embedding, layer_norm, conv1d, VarBuilder, Dropout, linear, Linear, ops::softmax};

pub struct PointWiseFeedForward {
    conv1: candle_nn::Conv1d,
    dropout1: Dropout,
    conv2: candle_nn::Conv1d,
    dropout2: Dropout,
}

impl PointWiseFeedForward {
    pub fn new(hidden_units: usize, dropout_rate: f32, vb: &VarBuilder) -> Result<Self> {
        let conv1 = conv1d(hidden_units, hidden_units, 1, Default::default(), vb.pp("conv1"))?;
        let dropout1 = Dropout::new(dropout_rate);
        let conv2 = conv1d(hidden_units, hidden_units, 1, Default::default(), vb.pp("conv2"))?;
        let dropout2 = Dropout::new(dropout_rate);
        Ok(Self { conv1, dropout1, conv2, dropout2 })
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let xs = xs.transpose(2, 1)?;
        let xs = self.conv1.forward(&xs)?;
        let xs = self.dropout1.forward(&xs, train)?;
        let xs = xs.relu()?;
        let xs = self.conv2.forward(&xs)?;
        let xs = self.dropout2.forward(&xs, train)?;
        xs.transpose(2, 1)
    }
}

pub struct MultiHeadAttention {
    n_head: usize,
    n_embd: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
}

impl MultiHeadAttention {
    pub fn new(vb: VarBuilder, n_head: usize, n_embd: usize) -> Result<Self> {
        let q_proj = linear(n_embd, n_embd, vb.pp("q_proj"))?;
        let k_proj = linear(n_embd, n_embd, vb.pp("k_proj"))?;
        let v_proj = linear(n_embd, n_embd, vb.pp("v_proj"))?;
        let out_proj = linear(n_embd, n_embd, vb.pp("out_proj"))?;
        Ok(Self {
            n_head,
            n_embd,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        // println!("Input shape: {:?}", x.shape());
        // println!("Expected embedding dim: {}", self.n_embd);
        // println!("Actual embedding dim: {}", n_embd);
        
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        
        // println!("Q shape: {:?}", q.shape());
        // println!("K shape: {:?}", k.shape());
        // println!("V shape: {:?}", v.shape());
        
        // Compute attention scores
        let scale = (n_embd as f32).powf(-0.5);
        
        // Reshape for multi-head attention
        let q = q.reshape((b_sz, seq_len, self.n_head, n_embd / self.n_head))?;
        let k = k.reshape((b_sz, seq_len, self.n_head, n_embd / self.n_head))?;
        let v = v.reshape((b_sz, seq_len, self.n_head, n_embd / self.n_head))?;
        
        // Transpose for attention computation
        let q = q.transpose(1, 2)?;  // [batch_size, n_head, seq_len, head_dim]
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        
        let attn = q.matmul(&k.transpose(2, 3)?)?;  // [batch_size, n_head, seq_len, seq_len]
        let attn = attn.broadcast_mul(&Tensor::new(scale, x.device())?)?;
        
        // Apply causal mask
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                mask_data[i * seq_len + j] = 0.0;
            }
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        let mask = Tensor::from_slice(&mask_data, (seq_len, seq_len), x.device())?;
        let mask = mask.unsqueeze(0)?.unsqueeze(0)?;  // [1, 1, seq_len, seq_len]
        let mask = mask.broadcast_as(&[b_sz, self.n_head, seq_len, seq_len])?;
        
        let attn = attn.broadcast_add(&mask)?;
        let attn = softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v)?;  // [batch_size, n_head, seq_len, head_dim]
        
        // Reshape back
        let out = out.transpose(1, 2)?;  // [batch_size, seq_len, n_head, head_dim]
        let out = out.reshape((b_sz, seq_len, n_embd))?;  // [batch_size, seq_len, n_embd]
        
        // Final projection
        self.out_proj.forward(&out)
    }
}

pub struct TransformerBlock {
    attn: MultiHeadAttention,
    mlp: MLP,
    ln1: LayerNorm,
    ln2: LayerNorm,
}

impl TransformerBlock {
    pub fn new(vb: VarBuilder, n_head: usize, n_embd: usize) -> Result<Self> {
        let attn = MultiHeadAttention::new(vb.pp("attn"), n_head, n_embd)?;
        let mlp = MLP::new(vb.pp("mlp"), n_embd)?;
        let ln1 = LayerNorm::new(vb.pp("ln1"), n_embd)?;
        let ln2 = LayerNorm::new(vb.pp("ln2"), n_embd)?;
        Ok(Self { attn, mlp, ln1, ln2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.ln1.forward(x)?;
        let x = self.attn.forward(&x, None)?;
        let x = residual.broadcast_add(&x)?;
        
        let residual = &x;
        let x = self.ln2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        residual.broadcast_add(&x)
    }
}

pub struct MLP {
    fc1: Linear,
    fc2: Linear,
}

impl MLP {
    pub fn new(vb: VarBuilder, n_embd: usize) -> Result<Self> {
        let fc1 = linear(n_embd, 4 * n_embd, vb.pp("fc1"))?;
        let fc2 = linear(4 * n_embd, n_embd, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.gelu()?;
        self.fc2.forward(&x)
    }
}

pub struct LayerNorm {
    inner: candle_nn::LayerNorm,
}

impl LayerNorm {
    pub fn new(vb: VarBuilder, n_embd: usize) -> Result<Self> {
        let inner = layer_norm(n_embd, 1e-5, vb)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

pub struct SASRec {
    item_emb: candle_nn::Embedding,
    user_emb: candle_nn::Embedding,
    pos_emb: candle_nn::Embedding,
    emb_dropout: Dropout,
    attn_layers: Vec<MultiHeadAttention>,
    ffn_layers: Vec<PointWiseFeedForward>,
    layer_norm: candle_nn::LayerNorm,
    output: Linear,
    dev: Device,
    hidden_units: usize,
}

impl SASRec {
    pub fn new(n_items: usize, n_users: usize, args: &Args, vb: &VarBuilder) -> Result<Self> {
        // The input IDs are already 0-based (0 to n_items-1), and 0 is used for padding
        // So we need to add 1 to the vocabulary size to account for padding
        let item_emb = embedding(n_items, args.hidden_units, vb.pp("item_emb"))?;
        let user_emb = embedding(n_users, args.hidden_units, vb.pp("user_emb"))?;
        let pos_emb = embedding(args.maxlen, args.hidden_units, vb.pp("pos_emb"))?;
        let emb_dropout = Dropout::new(args.dropout_rate);
        
        let mut attn_layers = Vec::new();
        for i in 0..args.num_blocks {
            let attn_layer = MultiHeadAttention::new(
                vb.pp(&format!("attn_layer_{}", i)),
                args.num_heads,
                args.hidden_units
            )?;
            attn_layers.push(attn_layer);
        }
        
        let mut ffn_layers = Vec::new();
        for i in 0..args.num_blocks {
            let ffn = PointWiseFeedForward::new(
                args.hidden_units,
                args.dropout_rate,
                &vb.pp(&format!("ffn_layer_{}", i))
            )?;
            ffn_layers.push(ffn);
        }
        
        let layer_norm = layer_norm(args.hidden_units, 1e-8, vb.pp("layer_norm"))?;
        let output = linear(args.hidden_units, n_items, vb.pp("output"))?;
        
        Ok(Self {
            item_emb,
            user_emb,
            pos_emb,
            emb_dropout,
            attn_layers,
            ffn_layers,
            layer_norm,
            output,
            dev: args.device.clone(),
            hidden_units: args.hidden_units,
        })
    }

    pub fn forward(&self, seqs: &Tensor, user: &Tensor, train: bool) -> Result<Tensor> {
        // Get sequence length and create position indices
        let seq_len = seqs.dim(1)?;
        let pos = Tensor::arange(0, seq_len as u32, &self.dev)?;
        
        // Create attention mask (1 for real tokens, 0 for padding)
        let mask = seqs.ne(0u32)?.to_dtype(DType::F32)?;  // [batch_size, seq_len]
        
        // Get embeddings
        let seqs = self.item_emb.forward(seqs)?;  // [batch_size, seq_len, hidden_units]
        let pos_emb = self.pos_emb.forward(&pos)?;  // [seq_len, hidden_units]
        let user = self.user_emb.forward(user)?;  // [batch_size, hidden_units]
        
        // Add position embeddings
        let pos_emb = pos_emb.unsqueeze(0)?;  // [1, seq_len, hidden_units]
        let seqs = (seqs + pos_emb)?;
        
        // Apply dropout
        let seqs = self.emb_dropout.forward(&seqs, train)?;
        
        // Expand user embedding to match sequence length
        let user = user.unsqueeze(1)?.expand(&[1, seq_len, self.hidden_units])?;
        
        // Add user embedding to sequence embeddings
        let mut seqs = (seqs + user)?;
        
        // Apply transformer blocks
        for (attn_layer, ffn_layer) in self.attn_layers.iter().zip(self.ffn_layers.iter()) {
            // Self-attention
            let mha_outputs = attn_layer.forward(&seqs, Some(&mask))?;
            seqs = (seqs + mha_outputs)?;
            
            // Feed-forward
            let fwd = ffn_layer.forward(&seqs, train)?;
            seqs = (seqs + fwd)?;
        }
        
        // Final layer normalization
        let seqs = self.layer_norm.forward(&seqs)?;
        
        // Final projection to item space
        self.output.forward(&seqs)
    }
}

#[derive(Clone)]
pub struct Args {
    pub hidden_units: usize,
    pub num_heads: usize,
    pub num_blocks: usize,
    pub dropout_rate: f32,
    pub maxlen: usize,
    pub device: Device,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            hidden_units: 64,
            num_heads: 1,
            num_blocks: 2,
            dropout_rate: 0.2,
            maxlen: 200,
            device: Device::Cpu,
        }
    }
}
