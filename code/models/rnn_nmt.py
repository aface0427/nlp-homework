"""
RNN-based Neural Machine Translation Model with Attention
Implements Seq2Seq with GRU/LSTM and various attention mechanisms
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import config as config


class Encoder(nn.Module):
    """
    Encoder for Seq2Seq model using GRU or LSTM
    Two unidirectional layers as per assignment requirement
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        cell_type: str = "GRU",
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        
        # Create RNN layer (GRU or LSTM)
        rnn_class = nn.GRU if cell_type == "GRU" else nn.LSTM
        self.rnn = rnn_class(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # Unidirectional as per requirement
        )
    
    def forward(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src: Source sequences [batch_size, seq_len]
            src_lengths: Lengths of source sequences [batch_size]
        
        Returns:
            outputs: Encoder outputs [batch_size, seq_len, hidden_dim]
            hidden: Final hidden state
        """
        # Embed and apply dropout
        embedded = self.dropout(self.embedding(src))  # [batch, seq_len, embed_dim]
        
        # Pack padded sequence if lengths provided
        if src_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        outputs, hidden = self.rnn(embedded)
        
        # Unpack if packed
        if src_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        return outputs, hidden


class Attention(nn.Module):
    """
    Attention mechanism supporting dot-product, multiplicative, and additive attention
    """
    
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        attention_type: str = "additive"
    ):
        super().__init__()
        
        self.attention_type = attention_type
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        
        if attention_type == "dot":
            # Dot-product attention: requires encoder_dim == decoder_dim
            assert encoder_dim == decoder_dim, "Dot attention requires equal dimensions"
        
        elif attention_type == "multiplicative":
            # Multiplicative (general) attention: score = h_t^T W h_s
            self.W = nn.Linear(encoder_dim, decoder_dim, bias=False)
        
        elif attention_type == "additive":
            # Additive (Bahdanau) attention: score = v^T tanh(W1*h_s + W2*h_t)
            self.W1 = nn.Linear(encoder_dim, decoder_dim, bias=False)
            self.W2 = nn.Linear(decoder_dim, decoder_dim, bias=False)
            self.v = nn.Linear(decoder_dim, 1, bias=False)
        
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
    
    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden: Decoder hidden state [batch_size, decoder_dim]
            encoder_outputs: Encoder outputs [batch_size, src_len, encoder_dim]
            src_mask: Mask for source padding [batch_size, src_len]
        
        Returns:
            context: Context vector [batch_size, encoder_dim]
            attention_weights: Attention weights [batch_size, src_len]
        """
        batch_size, src_len, _ = encoder_outputs.shape
        
        if self.attention_type == "dot":
            # [batch, src_len, encoder_dim] @ [batch, decoder_dim, 1] -> [batch, src_len, 1]
            scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        
        elif self.attention_type == "multiplicative":
            # Transform encoder outputs: [batch, src_len, decoder_dim]
            transformed = self.W(encoder_outputs)
            # [batch, src_len, decoder_dim] @ [batch, decoder_dim, 1] -> [batch, src_len]
            scores = torch.bmm(transformed, decoder_hidden.unsqueeze(2)).squeeze(2)
        
        elif self.attention_type == "additive":
            # [batch, src_len, decoder_dim]
            encoder_proj = self.W1(encoder_outputs)
            # [batch, 1, decoder_dim]
            decoder_proj = self.W2(decoder_hidden).unsqueeze(1)
            # [batch, src_len, decoder_dim] -> [batch, src_len, 1] -> [batch, src_len]
            scores = self.v(torch.tanh(encoder_proj + decoder_proj)).squeeze(2)
        
        # Apply mask (set padded positions to -inf)
        if src_mask is not None:
            scores = scores.masked_fill(src_mask == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=1)
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights


class Decoder(nn.Module):
    """
    Decoder for Seq2Seq model with attention mechanism
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        cell_type: str = "GRU",
        attention_type: str = "additive",
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        
        # Attention mechanism
        self.attention = Attention(hidden_dim, hidden_dim, attention_type)
        
        # RNN layer
        # Input: embedding + context vector
        rnn_class = nn.GRU if cell_type == "GRU" else nn.LSTM
        self.rnn = rnn_class(
            embedding_dim + hidden_dim,  # Concatenate embedding and context
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.fc_out = nn.Linear(hidden_dim * 2 + embedding_dim, vocab_size)
    
    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single decoding step
        
        Args:
            input_token: Input token [batch_size]
            hidden: Previous hidden state
            encoder_outputs: Encoder outputs [batch_size, src_len, hidden_dim]
            src_mask: Source mask [batch_size, src_len]
        
        Returns:
            output: Output logits [batch_size, vocab_size]
            hidden: New hidden state
            attention_weights: Attention weights [batch_size, src_len]
        """
        # Get embedding
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))  # [batch, 1, embed_dim]
        
        # Get the top layer hidden state for attention
        if self.cell_type == "LSTM":
            h_for_attn = hidden[0][-1]  # Last layer hidden state
        else:
            h_for_attn = hidden[-1]  # Last layer hidden state
        
        # Compute attention
        context, attention_weights = self.attention(h_for_attn, encoder_outputs, src_mask)
        
        # Concatenate embedding and context
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        
        # Run through RNN
        output, hidden = self.rnn(rnn_input, hidden)
        
        # Concatenate output, context, and embedding for prediction
        output = output.squeeze(1)
        prediction_input = torch.cat([output, context, embedded.squeeze(1)], dim=1)
        output = self.fc_out(prediction_input)
        
        return output, hidden, attention_weights


class Seq2SeqRNN(nn.Module):
    """
    Complete Seq2Seq model with encoder, decoder, and attention
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        cell_type: str = "GRU",
        attention_type: str = "additive",
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.encoder = Encoder(
            src_vocab_size, embedding_dim, hidden_dim,
            num_layers, dropout, cell_type, padding_idx
        )
        
        self.decoder = Decoder(
            tgt_vocab_size, embedding_dim, hidden_dim,
            num_layers, dropout, cell_type, attention_type, padding_idx
        )
        
        self.tgt_vocab_size = tgt_vocab_size
        self.cell_type = cell_type
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Forward pass with optional teacher forcing
        
        Args:
            src: Source sequences [batch_size, src_len]
            tgt: Target sequences [batch_size, tgt_len]
            src_mask: Source mask [batch_size, src_len]
            teacher_forcing_ratio: Probability of using teacher forcing
        
        Returns:
            outputs: Output logits [batch_size, tgt_len, vocab_size]
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        
        # Encode source
        src_lengths = src_mask.sum(dim=1) if src_mask is not None else None
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # Initialize output tensor
        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size, device=src.device)
        
        # First input is SOS token
        input_token = tgt[:, 0]
        
        for t in range(1, tgt_len):
            # Decode one step
            output, hidden, _ = self.decoder(input_token, hidden, encoder_outputs, src_mask)
            outputs[:, t] = output
            
            # Decide whether to use teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            
            if use_teacher_forcing:
                input_token = tgt[:, t]
            else:
                input_token = output.argmax(dim=1)
        
        return outputs
    
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        max_len: int = 100,
        sos_idx: int = 2,
        eos_idx: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy decoding
        
        Returns:
            decoded: Decoded token indices [batch_size, max_len]
            attention_weights: Attention weights for visualization
        """
        batch_size = src.size(0)
        device = src.device
        
        # Encode source
        src_lengths = src_mask.sum(dim=1) if src_mask is not None else None
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # Initialize
        decoded = torch.full((batch_size, max_len), eos_idx, dtype=torch.long, device=device)
        decoded[:, 0] = sos_idx
        input_token = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)
        
        all_attention_weights = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for t in range(1, max_len):
            output, hidden, attention_weights = self.decoder(
                input_token, hidden, encoder_outputs, src_mask
            )
            all_attention_weights.append(attention_weights)
            
            # Get predicted token
            predicted = output.argmax(dim=1)
            decoded[:, t] = predicted
            
            # Update finished status
            finished = finished | (predicted == eos_idx)
            if finished.all():
                break
            
            input_token = predicted
        
        attention_weights = torch.stack(all_attention_weights, dim=1) if all_attention_weights else None
        return decoded, attention_weights
    
    def beam_search_decode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        beam_size: int = 5,
        max_len: int = 100,
        length_penalty: float = 0.6,
        sos_idx: int = 2,
        eos_idx: int = 3
    ) -> torch.Tensor:
        """
        Beam search decoding
        
        Returns:
            decoded: Best decoded sequence [batch_size, max_len]
        """
        batch_size = src.size(0)
        device = src.device
        
        # For simplicity, process one sample at a time
        all_decoded = []
        
        for i in range(batch_size):
            single_src = src[i:i+1]
            single_mask = src_mask[i:i+1] if src_mask is not None else None
            
            decoded = self._beam_search_single(
                single_src, single_mask, beam_size, max_len, length_penalty, sos_idx, eos_idx
            )
            all_decoded.append(decoded)
        
        # Pad to same length
        max_decoded_len = max(len(d) for d in all_decoded)
        result = torch.full((batch_size, max_decoded_len), eos_idx, dtype=torch.long, device=device)
        for i, decoded in enumerate(all_decoded):
            result[i, :len(decoded)] = torch.tensor(decoded, device=device)
        
        return result
    
    def _beam_search_single(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        beam_size: int,
        max_len: int,
        length_penalty: float,
        sos_idx: int,
        eos_idx: int
    ):
        """Beam search for a single sample"""
        device = src.device
        
        # Encode - src is [1, src_len]
        # Don't use pack_padded_sequence to avoid length mismatch issues
        # encoder_outputs will have same length as src
        encoder_outputs, hidden = self.encoder(src, None)
        # encoder_outputs: [1, src_len, hidden_dim]
        
        # Make sure src_mask matches encoder_outputs length
        if src_mask is not None:
            src_len = encoder_outputs.size(1)
            src_mask = src_mask[:, :src_len]
        
        # Initialize beams
        # Each beam: (score, tokens, hidden, finished)
        beams = [(0.0, [sos_idx], hidden, False)]
        
        for _ in range(max_len - 1):
            all_candidates = []
            
            for score, tokens, hidden, finished in beams:
                if finished:
                    all_candidates.append((score, tokens, hidden, finished))
                    continue
                
                # Get last token - decoder expects [batch_size] shape
                input_token = torch.tensor([tokens[-1]], device=device)
                
                # Decode one step
                # Note: encoder_outputs is [1, src_len, hidden_dim]
                # src_mask is [1, src_len]
                output, new_hidden, _ = self.decoder(
                    input_token, 
                    hidden,
                    encoder_outputs,
                    src_mask
                )
                
                # Get top-k candidates
                log_probs = F.log_softmax(output[0], dim=-1)
                top_k_probs, top_k_indices = log_probs.topk(beam_size)
                
                for prob, idx in zip(top_k_probs.tolist(), top_k_indices.tolist()):
                    new_tokens = tokens + [idx]
                    # Apply length penalty
                    new_score = (score + prob) / (len(new_tokens) ** length_penalty)
                    new_finished = (idx == eos_idx)
                    all_candidates.append((new_score, new_tokens, new_hidden, new_finished))
            
            # Select top beams
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = all_candidates[:beam_size]
            
            # Check if all beams are finished
            if all(b[3] for b in beams):
                break
        
        # Return best beam
        best_tokens = beams[0][1]
        return best_tokens


def create_rnn_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    config_dict: dict = None
) -> Seq2SeqRNN:
    """Factory function to create RNN model"""
    if config_dict is None:
        config_dict = config.RNN_CONFIG
    
    return Seq2SeqRNN(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embedding_dim=config_dict["embedding_dim"],
        hidden_dim=config_dict["hidden_dim"],
        num_layers=config_dict["num_layers"],
        dropout=config_dict["dropout"],
        cell_type=config_dict["cell_type"],
        attention_type=config_dict["attention_type"],
        padding_idx=config.PAD_IDX
    )


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    src_len = 20
    tgt_len = 15
    src_vocab_size = 5000
    tgt_vocab_size = 8000
    
    model = create_rnn_model(src_vocab_size, tgt_vocab_size)
    
    # Create dummy data
    src = torch.randint(4, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(4, tgt_vocab_size, (batch_size, tgt_len))
    src_mask = torch.ones(batch_size, src_len)
    
    # Forward pass
    outputs = model(src, tgt, src_mask, teacher_forcing_ratio=0.5)
    print(f"Output shape: {outputs.shape}")  # Should be [batch_size, tgt_len, tgt_vocab_size]
    
    # Greedy decode
    decoded, attention = model.greedy_decode(src, src_mask, max_len=20)
    print(f"Decoded shape: {decoded.shape}")
    
    # Beam search decode
    decoded = model.beam_search_decode(src, src_mask, beam_size=3, max_len=20)
    print(f"Beam search decoded shape: {decoded.shape}")
