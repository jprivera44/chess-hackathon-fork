#%%
"""Transformer model."""

import dataclasses
import enum
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

@dataclasses.dataclass(kw_only=True)
class TransformerConfig:
    """Hyperparameters used in the Transformer architectures."""

    # The random seed for parameter initialization.
    seed: int = 1
    # The input vocabulary size.
    vocab_size: int
    # The output size (by default equal to the vocabulary size).
    output_size: int
    # The dimension of the first embedding.
    embedding_dim: int = 64
    # The number of multi-head attention layers.
    num_layers: int = 4
    # The number of heads per layer.
    num_heads: int = 8
    # The parameter initialization scale for the embeddings.
    emb_init_scale: float = 0.02
    # Maximum sequence length, useful for the LEARNED positional encodings.
    max_sequence_length: int
    # How much larger the hidden layer of the feedforward network should be
    # compared to the `embedding_dim`.
    widening_factor: int = 4

    def __post_init__(self):
        if self.output_size is None:
            self.output_size = self.vocab_size


class MultiHeadDotProductAttention(nn.Module):
    """Multi-head dot-product attention (Vaswani et al., 2017)."""

    def __init__(
        self,
        num_heads: int,
        num_hiddens_per_head: int,
    ) -> None:
        """Initializes the attention module.

        Args:
          num_heads: Number of heads to use.
          num_hiddens_per_head: Number of hidden neurons per head.
        """
        super().__init__()
        self._num_heads = num_heads
        self._num_hiddens_per_head = num_hiddens_per_head
        
        num_hiddens = self._num_hiddens_per_head * self._num_heads
        self.q_linear = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.k_linear = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.v_linear = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.output_linear = nn.Linear(num_hiddens, num_hiddens, bias=False)
        
        self.q_layer_norm = nn.LayerNorm(num_hiddens)
        self.k_layer_norm = nn.LayerNorm(num_hiddens)

    def forward(
        self,
        inputs_q: torch.Tensor,
        inputs_kv: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the output of the multi-head attention."""
        batch_size, sequence_length, embedding_size = inputs_q.shape

        q = self.q_linear(inputs_q)
        k = self.k_linear(inputs_kv)

        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)

        v = self.v_linear(inputs_kv)

        new_shape = (batch_size, -1, self._num_heads, self._num_hiddens_per_head)
        q = q.view(new_shape)
        k = k.view(new_shape)
        v = v.view(new_shape)

        attention = torch.einsum('bthd,bThd->bhtT', q, k)
        attention *= 1.0 / np.sqrt(self._num_hiddens_per_head)
        normalized_attention = F.softmax(attention, dim=-1)

        output = torch.einsum('bhtT,bThd->bthd', normalized_attention, v)
        output = output.reshape(batch_size, sequence_length, embedding_size)
        return self.output_linear(output)


def embed_sequences(
    sequences: torch.Tensor,
    config: TransformerConfig,
) -> torch.Tensor:
    """Returns embeddings for sequences of tokens."""
    embeddings_layer = nn.Embedding(
        num_embeddings=config.vocab_size,
        embedding_dim=config.embedding_dim,
    )
    nn.init.trunc_normal_(embeddings_layer.weight, std=config.emb_init_scale)
    
    embeddings = embeddings_layer(sequences)
    embeddings *= np.sqrt(config.embedding_dim)

    _, sequence_length = sequences.shape
    assert sequence_length <= config.max_sequence_length
    positions = torch.arange(sequence_length, device=sequences.device)
    pos_encodings = nn.Embedding(
        num_embeddings=config.max_sequence_length,
        embedding_dim=config.embedding_dim,
    )(positions)
    return embeddings + pos_encodings


def shift_right(sequences: torch.Tensor) -> torch.Tensor:
    """Right-shift the input by padding on the temporal axis."""
    bos_array = torch.zeros((sequences.shape[0], 1), dtype=torch.long, device=sequences.device)
    padded_sequences = torch.cat([bos_array, sequences], dim=1)
    return padded_sequences[:, :-1]


class MLPBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        ffn_dim = config.embedding_dim * config.widening_factor
        self.linear1 = nn.Linear(config.embedding_dim, ffn_dim, bias=False)
        self.linear2 = nn.Linear(config.embedding_dim, ffn_dim, bias=False)
        self.linear3 = nn.Linear(ffn_dim, config.embedding_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        split_1 = self.linear1(inputs)
        split_2 = self.linear2(inputs)
        gate_output = F.silu(split_1) * split_2
        return self.linear3(gate_output)


class TransformerDecoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Initialize embedding layers
        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
        )
        nn.init.trunc_normal_(self.token_embedding.weight, std=config.emb_init_scale)
        
        self.position_embedding = nn.Embedding(
            num_embeddings=config.max_sequence_length,
            embedding_dim=config.embedding_dim,
        )      

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiHeadDotProductAttention(
                    num_heads=config.num_heads,
                    num_hiddens_per_head=config.embedding_dim // config.num_heads,
                ),
                'mlp': MLPBlock(config),
                'ln1': nn.LayerNorm(config.embedding_dim),
                'ln2': nn.LayerNorm(config.embedding_dim),
            })
            for _ in range(config.num_layers)
        ])
        
        self.final_ln = nn.LayerNorm(config.embedding_dim)
        self.output_linear = nn.Linear(config.embedding_dim, config.output_size)
    
    def embed_sequences(self, sequences: torch.Tensor) -> torch.Tensor:
        """Returns embeddings for sequences of tokens."""
        embeddings = self.token_embedding(sequences)
        embeddings *= np.sqrt(self.config.embedding_dim)

        _, sequence_length = sequences.shape
        assert sequence_length <= self.config.max_sequence_length
        positions = torch.arange(sequence_length, device=sequences.device)
        pos_encodings = self.position_embedding(positions)
        return embeddings + pos_encodings

    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        inputs = shift_right(targets)
        embeddings = embed_sequences(inputs, self.config)

        h = embeddings
        for layer in self.layers:
            assert type(layer) == nn.ModuleDict
            attention_input = layer['ln1'](h)
            attention = layer['attention'](attention_input, attention_input)
            h = h + attention

            mlp_input = layer['ln2'](h)
            mlp_output = layer['mlp'](mlp_input)
            h = h + mlp_output

        h = self.final_ln(h)
        logits = self.output_linear(h)
        return F.log_softmax(logits, dim=-1)
    
