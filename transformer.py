import torch
import torch.nn as nn

from MultiHeadAttn import MultiHeadAttention
from Feedforward import FeedForward
from ResidualConnection import ResidualConnection
from Embedding import InputEmbedding
from PositionalEncoding import PositionalEncoding

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention, feedforward_block: FeedForward, dropout_p: float):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feedforward_block = feedforward_block
        self.dropout = nn.Dropout(dropout_p)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout_p) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feedforward_block)

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        
        return x
    

class DecoderBlock(nn.Module):

    def __init__(self, self_attn_block: MultiHeadAttention, cross_attn_block: MultiHeadAttention, feedforward_block: FeedForward, dropout_p: float) -> None:
        super().__init__()

        self.self_attention_block = self_attn_block
        self.cross_attention_block = cross_attn_block
        self.feedforward_block = feedforward_block
        self.dropout = nn.Dropout(dropout_p)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout_p) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feedforward_block)

        return x

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):

    def __init__(self,
                    encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding,
                    src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection: ProjectionLayer 
                 ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size:int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, n_heads: int = 8, dropout_p: int = 0.1, d_ff:int = 2048) -> Transformer:
    
    # Create embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size) 
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # Create positional encoding layers
    src_pos_embed = PositionalEncoding(d_model, src_seq_len, dropout_p) 
    tgt_pos_embed = PositionalEncoding(d_model, tgt_seq_len, dropout_p)

    # Create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attn_block = MultiHeadAttention(d_model, n_heads, dropout_p)
        ff_block = FeedForward(d_model, d_ff, dropout_p)
        encoder_block = EncoderBlock(encoder_self_attn_block, ff_block, dropout_p)
        encoder_blocks.append(encoder_block)


    # Create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attn_block = MultiHeadAttention(d_model, n_heads, dropout_p)
        decoder_x_attn_block = MultiHeadAttention(d_model, n_heads, dropout_p)
        ff_block = FeedForward(d_model, d_ff, dropout_p)
        decoder_block = DecoderBlock(decoder_self_attn_block, decoder_x_attn_block, ff_block, dropout_p)
        decoder_blocks.append(decoder_block)
    
    # Create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create final projection layer
    proj_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create Transformer
    transformer = Transformer(encoder, decoder, tgt_embed, src_embed, tgt_pos_embed, src_pos_embed, proj_layer)

    # Initialize params using Xavier init
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer