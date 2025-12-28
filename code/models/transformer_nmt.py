"""
Transformer-based Neural Machine Translation Model
Implements Transformer from scratch with configurable position encoding (absolute/relative)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import config as config


class PositionalEncoding(nn.Module):
    """
    Absolute positional encoding using sine and cosine functions
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding using learnable embeddings
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        
        # Learnable relative position embeddings
        # We need embeddings for distances from -max_len to +max_len
        self.rel_pos_embedding = nn.Embedding(2 * max_len + 1, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        For relative PE, we don't add anything to the input embeddings directly.
        The relative information is added in the attention mechanism.
        """
        return self.dropout(x)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    """
    
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return self.weight * (x / rms)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism with support for Relative Positional Encoding
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        use_relative_pos: bool = False,
        max_len: int = 512
    ):
        super().__init__()
        
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.use_relative_pos = use_relative_pos
        self.max_len = max_len
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        if use_relative_pos:
            # Relative position embeddings for keys and values
            # Shared across heads, but we project them to d_k
            self.rel_k = nn.Embedding(2 * max_len + 1, self.d_k)
            self.rel_v = nn.Embedding(2 * max_len + 1, self.d_k)
    
    def _get_relative_positions(self, tgt_len: int, src_len: int, device: torch.device) -> torch.Tensor:
        """Generate relative position indices"""
        tgt_pos = torch.arange(tgt_len, device=device).unsqueeze(1)
        src_pos = torch.arange(src_len, device=device).unsqueeze(0)
        rel_pos = tgt_pos - src_pos + self.max_len
        rel_pos = rel_pos.clamp(0, 2 * self.max_len)
        return rel_pos

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, tgt_len, d_model]
            key: [batch_size, src_len, d_model]
            value: [batch_size, src_len, d_model]
            mask: [batch_size, 1, 1, src_len] or [batch_size, 1, tgt_len, src_len]
        
        Returns:
            output: [batch_size, tgt_len, d_model]
            attention_weights: [batch_size, nhead, tgt_len, src_len]
        """
        batch_size = query.size(0)
        tgt_len = query.size(1)
        src_len = key.size(1)
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, tgt_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, src_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, src_len, self.nhead, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        # [batch, nhead, tgt_len, d_k] @ [batch, nhead, d_k, src_len] -> [batch, nhead, tgt_len, src_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add relative position bias if enabled
        if self.use_relative_pos:
            rel_pos = self._get_relative_positions(tgt_len, src_len, query.device)
            # [tgt_len, src_len, d_k]
            rel_k_embed = self.rel_k(rel_pos)
            
            # We need to add (Q * rel_k^T) to scores
            # Q: [batch, nhead, tgt_len, d_k]
            # rel_k_embed: [tgt_len, src_len, d_k]
            # Result should be [batch, nhead, tgt_len, src_len]
            
            # Reshape Q for einsum: [batch, nhead, tgt_len, d_k]
            # Reshape rel_k for einsum: [tgt_len, src_len, d_k]
            # Output: [batch, nhead, tgt_len, src_len]
            rel_scores = torch.einsum('bhqd,qkd->bhqk', Q, rel_k_embed)
            
            scores = scores + rel_scores / math.sqrt(self.d_k)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply to values
        output = torch.matmul(attention_weights, V)
        
        # Add relative position values if enabled
        if self.use_relative_pos:
            rel_pos = self._get_relative_positions(tgt_len, src_len, query.device)
            rel_v_embed = self.rel_v(rel_pos)  # [tgt_len, src_len, d_k]
            
            # weights: [batch, nhead, tgt_len, src_len]
            # rel_v: [tgt_len, src_len, d_k]
            # output: [batch, nhead, tgt_len, d_k]
            rel_output = torch.einsum('bhqk,qkd->bhqd', attention_weights, rel_v_embed)
            output = output + rel_output
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)
        
        # Final linear projection
        output = self.W_o(output)
        
        return output, attention_weights


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    """
    
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        use_relative_pos: bool = False,
        max_len: int = 512,
        norm_type: str = "layernorm"
    ):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout, use_relative_pos, max_len)
        self.ff = FeedForward(d_model, dim_feedforward, dropout)
        
        if norm_type == "rmsnorm":
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_output, _ = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        
        # Feed-forward with residual
        ff_output = self.ff(src)
        src = src + self.dropout(ff_output)
        src = self.norm2(src)
        
        return src


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        use_relative_pos: bool = False,
        max_len: int = 512,
        norm_type: str = "layernorm"
    ):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout, use_relative_pos, max_len)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout, False, max_len) # Cross attn usually doesn't use relative pos of src-tgt
        self.ff = FeedForward(d_model, dim_feedforward, dropout)
        
        if norm_type == "rmsnorm":
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
            self.norm3 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_output, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout(attn_output)
        tgt = self.norm1(tgt)
        
        # Cross-attention with residual
        attn_output, _ = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = tgt + self.dropout(attn_output)
        tgt = self.norm2(tgt)
        
        # Feed-forward with residual
        ff_output = self.ff(tgt)
        tgt = tgt + self.dropout(ff_output)
        tgt = self.norm3(tgt)
        
        return tgt


class TransformerNMT(nn.Module):
    """
    Complete Transformer model for Neural Machine Translation
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        position_encoding: str = "absolute",
        norm_type: str = "layernorm",
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.tgt_vocab_size = tgt_vocab_size
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=padding_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=padding_idx)
        
        # Position encoding
        self.position_encoding_type = position_encoding
        use_relative_pos = (position_encoding == "relative")
        
        if position_encoding == "absolute":
            self.src_pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
            self.tgt_pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        elif position_encoding == "relative":
            self.src_pos_encoding = RelativePositionalEncoding(d_model, max_seq_len, dropout)
            self.tgt_pos_encoding = RelativePositionalEncoding(d_model, max_seq_len, dropout)
        else: # none
            self.src_pos_encoding = nn.Dropout(dropout)
            self.tgt_pos_encoding = nn.Dropout(dropout)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, use_relative_pos, max_seq_len, norm_type
            ) for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout, use_relative_pos, max_seq_len, norm_type
            ) for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float(0)).masked_fill(mask == 0, float(1))
        # In our custom attention, mask=1 means keep, mask=0 means ignore
        # But wait, let's check MultiHeadAttention implementation.
        # scores.masked_fill(mask == 0, float('-inf'))
        # So mask should be 1 for keep, 0 for ignore.
        
        # triu with diagonal=1 gives upper triangle (excluding diagonal) as 1s.
        # We want lower triangle (including diagonal) to be 1 (keep), upper to be 0 (ignore).
        
        mask = torch.tril(torch.ones(sz, sz, device=device))
        return mask.unsqueeze(0).unsqueeze(0) # [1, 1, sz, sz]
    
    def create_masks(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create attention masks"""
        device = src.device
        
        # Source padding mask: [batch_size, 1, 1, src_len]
        # 1 for keep, 0 for ignore
        src_mask = (src != self.padding_idx).unsqueeze(1).unsqueeze(2)
        
        # Target padding mask
        tgt_pad_mask = (tgt != self.padding_idx).unsqueeze(1).unsqueeze(2)
        
        # Causal mask for decoder
        tgt_len = tgt.size(1)
        causal_mask = self.generate_square_subsequent_mask(tgt_len, device)
        
        # Combine padding and causal masks
        tgt_mask = tgt_pad_mask & causal_mask.bool()
        
        # Memory mask (for cross-attention)
        memory_mask = src_mask
        
        return src_mask, tgt_mask, memory_mask
    
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """Encode source sequence"""
        # Embed and add positional encoding
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.src_pos_encoding(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory_mask: torch.Tensor
    ) -> torch.Tensor:
        """Decode target sequence"""
        # Embed and add positional encoding
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.tgt_pos_encoding(x)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        
        return x
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            src: Source sequences [batch_size, src_len]
            tgt: Target sequences [batch_size, tgt_len]
        
        Returns:
            output: Logits [batch_size, tgt_len, tgt_vocab_size]
        """
        # Create masks if not provided
        if src_mask is None or tgt_mask is None:
            src_mask, tgt_mask, memory_mask = self.create_masks(src, tgt)
        
        # Encode
        memory = self.encode(src, src_mask)
        
        # Decode
        output = self.decode(tgt, memory, tgt_mask, memory_mask)
        
        # Project to vocabulary
        output = self.fc_out(output)
        
        return output
    
    def greedy_decode(
        self,
        src: torch.Tensor,
        max_len: int = 100,
        sos_idx: int = 2,
        eos_idx: int = 3
    ) -> torch.Tensor:
        """
        Greedy decoding
        """
        batch_size = src.size(0)
        device = src.device
        
        # Create source mask
        src_mask = (src != self.padding_idx).unsqueeze(1).unsqueeze(2)
        
        # Encode source
        memory = self.encode(src, src_mask)
        
        # Initialize target with SOS
        tgt = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_len - 1):
            # Create target mask
            tgt_len = tgt.size(1)
            tgt_pad_mask = (tgt != self.padding_idx).unsqueeze(1).unsqueeze(2)
            causal_mask = self.generate_square_subsequent_mask(tgt_len, device)
            tgt_mask = tgt_pad_mask & causal_mask.bool()
            
            # Decode
            output = self.decode(tgt, memory, tgt_mask, src_mask)
            
            # Get last token prediction
            logits = self.fc_out(output[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Append to target
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Update finished status
            finished = finished | (next_token.squeeze(-1) == eos_idx)
            if finished.all():
                break
        
        return tgt


def create_transformer_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    config_dict: dict = None
) -> TransformerNMT:
    """Factory function to create Transformer model"""
    if config_dict is None:
        config_dict = config.TRANSFORMER_CONFIG
    
    return TransformerNMT(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config_dict["d_model"],
        nhead=config_dict["nhead"],
        num_encoder_layers=config_dict["num_encoder_layers"],
        num_decoder_layers=config_dict["num_decoder_layers"],
        dim_feedforward=config_dict["dim_feedforward"],
        dropout=config_dict["dropout"],
        max_seq_len=config_dict["max_seq_len"],
        position_encoding=config_dict["position_encoding"],
        norm_type=config_dict["norm_type"],
        padding_idx=config.PAD_IDX
    )


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    src_len = 20
    tgt_len = 15
    src_vocab_size = 5000
    tgt_vocab_size = 8000
    
    model = create_transformer_model(src_vocab_size, tgt_vocab_size)
    
    # Create dummy data
    src = torch.randint(4, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(4, tgt_vocab_size, (batch_size, tgt_len))
    
    # Forward pass
    outputs = model(src, tgt)
    print(f"Output shape: {outputs.shape}")  # Should be [batch_size, tgt_len, tgt_vocab_size]
    
    # Greedy decode
    decoded = model.greedy_decode(src, max_len=20)
    print(f"Decoded shape: {decoded.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
