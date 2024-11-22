import torch
import torch.nn as nn

'''
Below is the implementation of the meta aggregator using multi-head attention. Meta aggregator is used to 
refine the aggregated embedding of high-order neighbors.
'''

class Encoder(nn.Module):
    def __init__(self, embedding_size, num_blocks, num_heads, d_ff, dropout_rate):
        """
        Encoder module for multi-head attention and feedforward transformations.

        Args:
            embedding_size (int): Size of embeddings.
            num_blocks (int): Number of attention blocks.
            num_heads (int): Number of attention heads.
            d_ff (int): Hidden layer size in feedforward network.
            dropout_rate (float): Dropout rate.
        """
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        # Define multi-head attention and feedforward layers for each block
        self.attention_blocks = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embedding_size, num_heads=num_heads, dropout=dropout_rate)
            for _ in range(num_blocks)
        ])
        self.feedforward_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_size, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, embedding_size),
                nn.Dropout(dropout_rate)
            )
            for _ in range(num_blocks)
        ])
        self.layer_norms_attention = nn.ModuleList([nn.LayerNorm(embedding_size) for _ in range(num_blocks)])
        self.layer_norms_ff = nn.ModuleList([nn.LayerNorm(embedding_size) for _ in range(num_blocks)])

    def forward(self, input):
        """
        Forward pass for the encoder.

        Args:
            input (Tensor): Input tensor of shape [batch_size, num_neighbors, embedding_size].

        Returns:
            enc (Tensor): Output tensor of shape [batch_size, num_neighbors, embedding_size].
        """
        # Scale the input embeddings
        enc = input * (self.embedding_size ** 0.5)  # [b, n, e]

        # Process through each block
        for i in range(self.num_blocks):
            # Multi-head self-attention
            enc_transposed = enc.permute(1, 0, 2)  # Convert to [n, b, e] for PyTorch MultiheadAttention
            attn_output, _ = self.attention_blocks[i](enc_transposed, enc_transposed, enc_transposed)
            attn_output = attn_output.permute(1, 0, 2)  # Convert back to [b, n, e]
            # Add and normalize
            enc = self.layer_norms_attention[i](enc + attn_output)

            # Feedforward
            ff_output = self.feedforward_blocks[i](enc)
            # Add and normalize
            enc = self.layer_norms_ff[i](enc + ff_output)

        return enc  # [b, n, e]
