import torch
import einx
import math

from cs336_basics.utils import scaled_dot_product_attention


class Linear(torch.nn.Module):
    def __init__(self, in_features: int,
                 out_features: int,
                 device: torch.device|None=None,
                 dtype: torch.dtype|None=None):
        super().__init__()
        sigma = math.sqrt(2 / (in_features + out_features))
        self.weight = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(torch.empty(out_features, in_features, device=device, dtype=dtype),
                                        0, sigma, -3 * sigma, 3 * sigma)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einx.dot(
            '... d_in, d_out d_in -> ... d_out',
            x, self.weight
        )

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int,
                 embedding_dim: int,
                 device: torch.device|None=None,
                 dtype: torch.dtype|None=None):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
                                        0, 1, -3, 3)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
         return self.weight[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones(d_model, dtype=dtype, device=device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        d_model = x.size(-1)
        square_sum = einx.sum('... [d_model]', x * x)
        rms = torch.sqrt(square_sum / d_model + self.eps)
        weights = einx.divide(
            '[...] d_model', x, rms
        )
        rmsnorm = einx.multiply('... [d_model]', weights, self.weight)
        return rmsnorm.to(in_type)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inter_linear_trans = self.w1.forward(x)
        silu = inter_linear_trans * torch.sigmoid(inter_linear_trans)

        linear_trans = self.w3.forward(x)
        element_prod = silu * linear_trans
        return self.w2.forward(element_prod)

class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device|None=None):
        super().__init__()
        numerator = torch.arange(0, max_seq_len, device=device)
        denominator = theta ** (2 * torch.arange(0, d_k // 2, device=device) / d_k)
        theta = einx.divide(
            'row, col -> row col', numerator, denominator
        )
        rotations_x = torch.stack([torch.cos(theta), -torch.sin(theta)], dim=-1)
        rotations_y = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)
        rotations = torch.stack([rotations_x, rotations_y], dim=-1)
        self.register_buffer("rotations", rotations, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        dim_except_last = x.size()[:-1]
        x = x.view(*dim_except_last, -1, 2)
        token_positions = token_positions.expand(*dim_except_last)
        rotations = self.rotations[token_positions, :, :]
        x_rotated = einx.dot('... len d_model before_rot, ... len d_model before_rot after_rot -> ... len d_model after_rot', x, rotations)
        return x_rotated.view(*dim_except_last, -1)


class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int|None, max_seq_len: int|None=None, theta: float|None=None, device: torch.device|None=None, dtype=None):
        """
        Args:
            d_model: dimension of embedding
            num_heads: number of attention heads, if None, head dimension is removed
            max_seq_len: maximum sequence length
            theta: rotation parameter for RoPE
            device: device of model
            dtype: data type of model
        """
        super().__init__()
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.device=device
        self.num_heads = num_heads
        if max_seq_len is None or theta is None:
            self.RoPE = None
        else:
            self.RoPE = RoPE(theta=theta, d_k=d_model if num_heads is None else d_model // num_heads,
                             max_seq_len=max_seq_len, device=device)

    def forward(self, x, token_positions: torch.Tensor|None=None):
        seq_len = x.size(-2)
        Q = self.q_proj(x)
        if self.num_heads is not None:
            Q = einx.rearrange('... seq_len (h d_k) -> ... h seq_len d_k', Q, h=self.num_heads)
        K = self.k_proj(x)
        if self.num_heads is not None:
            K = einx.rearrange('... seq_len (h d_k) -> ... h seq_len d_k', K, h=self.num_heads)
        V = self.v_proj(x)
        if self.num_heads is not None:
            V = einx.rearrange('... seq_len (h d_v) -> ... h seq_len d_v', V, h=self.num_heads)

        if token_positions is not None:
            Q = self.RoPE.forward(Q, token_positions)
            K = self.RoPE.forward(K, token_positions)


        masks = torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device)
        masks = torch.triu(masks).transpose(0, 1)
        masks = einx.logical_and('..., q_len k_len -> ... q_len k_len',
                         torch.ones(K.size()[:-2], dtype=torch.bool, device=self.device), masks)

        output = scaled_dot_product_attention(Q, K, V, masks)
        if self.num_heads is not None:
            output = einx.rearrange('... h seq_len d_k -> ... seq_len (h d_k)', output)

        return self.output_proj.forward(output)

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int|None, d_ff: int, eps: float=1e-5,
                 max_seq_len: int|None=None, theta: float|None=None,
                 dtype: torch.dtype|None=None, device: torch.device|None=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, eps=eps, dtype=dtype, device=device)
        self.ln2 = RMSNorm(d_model, eps=eps, dtype=dtype, device=device)
        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len=max_seq_len, theta=theta,
                                           dtype=dtype, device=device)
        self.ffn = SwiGLU(d_model, d_ff, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor|None=None):
        normed_x = self.ln1.forward(x)
        seq_len = x.size(-2)
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        attention_x = self.attn.forward(normed_x, token_positions)
        processed_x = x + attention_x

        normed_x = self.ln2.forward(processed_x)
        feed_forward_x = self.ffn.forward(normed_x)
        return feed_forward_x + processed_x

class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int|None, d_ff: int,
                 context_length: int, theta: float, eps: float=1e-5,  
                 dtype: torch.dtype|None=None, device: torch.device|None=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, dtype=dtype, device=device)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, eps=eps, max_seq_len=context_length,
                             theta=theta, dtype=dtype, device=device)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, eps=eps, dtype=dtype, device=device)
        self.lm_head = Linear(d_model, vocab_size, dtype=dtype, device=device)
    
    def forward(self, x: torch.Tensor):
        x_embedded = self.token_embeddings(x)
        for layer in self.layers:
            x_embedded = layer(x_embedded)
        normed_embedding = self.ln_final.forward(x_embedded)
        output = self.lm_head.forward(normed_embedding)
        return output



