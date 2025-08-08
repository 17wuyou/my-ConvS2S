# conv_s2s/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionLayer(nn.Module):
    """
    实现了论文3.3节的多步注意力机制
    """
    def __init__(self, decoder_hidden_dim, embed_dim):
        super().__init__()
        # 对应公式(1)中的 Wdˡ，用于将解码器状态投射到与词嵌入相同的维度以进行相加
        self.query_projection = nn.Linear(decoder_hidden_dim, embed_dim, bias=False)

    def forward(self, decoder_state, prev_word_embedding, encoder_out_dict):
        encoder_output = encoder_out_dict['encoder_output']      # zⱼᵘ
        encoder_embedding = encoder_out_dict['encoder_embedding']# eⱼ
        
        # decoder_state (hᵢˡ) 是 (B, C, T), prev_word_embedding (gᵢ) 是 (B, T, C)
        decoder_state_t = decoder_state.transpose(1, 2)

        # 1. 计算查询向量 dᵢˡ (公式1)
        query = self.query_projection(decoder_state_t) + prev_word_embedding

        # 2. 计算注意力权重 aᵢⱼˡ (点积注意力)
        attn_scores = torch.matmul(query, encoder_output.transpose(1, 2))
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 3. 计算上下文向量 cᵢˡ (公式2)
        value_vectors = encoder_output + encoder_embedding
        context_vector = torch.matmul(attn_weights, value_vectors)
        
        # 将上下文向量的方差缩放回原始大小 (3.4节)
        # 假设注意力均匀分布，加权和会使方差缩小m倍，这里乘sqrt(m)来补偿
        context_vector = context_vector * math.sqrt(encoder_output.size(1))

        # 将 (B, T, C) 转回 (B, C, T) 以匹配卷积块的输入
        return context_vector.transpose(1, 2)

class EncoderConvolutionalBlock(nn.Module):
    """
    编码器中的卷积块
    """
    def __init__(self, hidden_dim, kernel_size, dropout):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size, padding=padding)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        residual = x
        x_conv = self.conv(x)
        x_glu = F.glu(x_conv, dim=1) # dim=1 是通道维度
        x_glu = self.dropout(x_glu)
        
        x = x_glu + residual
        x = x * math.sqrt(0.5) # 归一化残差连接后的方差
        return x

class DecoderConvolutionalBlock(nn.Module):
    """
    解码器中的卷积块，包含因果卷积和注意力
    """
    def __init__(self, hidden_dim, embed_dim, kernel_size, dropout):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size, padding=kernel_size - 1)
        self.attention = AttentionLayer(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        # 层归一化可以进一步稳定训练
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, prev_word_embedding, encoder_out_dict):
        residual = x

        # 1. 因果卷积
        x_conv = self.conv(x)
        # 移除右侧多余的 k-1 个输出以保持长度和因果性
        x_conv = x_conv[:, :, : -self.kernel_size + 1]
        x_glu = F.glu(x_conv, dim=1)
        x_glu = self.dropout(x_glu)

        # 2. 多步注意力
        attention_out = self.attention(x_glu, prev_word_embedding, encoder_out_dict)
        
        # 3. 将注意力和卷积输出结合
        x = x_glu + attention_out
        
        # 4. 残差连接
        x = x + residual
        x = x * math.sqrt(0.5)
        
        # PyTorch的LayerNorm期望 (B, T, C)
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)
        return x

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config['dropout']
        
        embed_dim = config['embed_dim']
        hidden_dim = config['hidden_dim']

        self.token_embedding = nn.Embedding(config['vocab_size_src'], embed_dim)
        self.position_embedding = nn.Embedding(config['max_seq_len'], embed_dim)

        self.embed_to_hidden = nn.Linear(embed_dim, hidden_dim)
        self.hidden_to_embed = nn.Linear(hidden_dim, embed_dim)

        self.conv_layers = nn.ModuleList([
            EncoderConvolutionalBlock(hidden_dim, config['encoder_kernel_size'], self.dropout)
            for _ in range(config['num_encoder_layers'])
        ])

    def forward(self, src_tokens):
        positions = torch.arange(src_tokens.size(1), device=src_tokens.device).unsqueeze(0)
        
        token_embed = self.token_embedding(src_tokens)
        pos_embed = self.position_embedding(positions)
        x = token_embed + pos_embed
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        encoder_embedding = x
        x = self.embed_to_hidden(x)
        x = x.transpose(1, 2)

        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = x.transpose(1, 2)
        x = self.hidden_to_embed(x)

        return {'encoder_output': x, 'encoder_embedding': encoder_embedding}

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config['dropout']
        embed_dim = config['embed_dim']
        hidden_dim = config['hidden_dim']

        self.token_embedding = nn.Embedding(config['vocab_size_tgt'], embed_dim)
        self.position_embedding = nn.Embedding(config['max_seq_len'], embed_dim)

        self.embed_to_hidden = nn.Linear(embed_dim, hidden_dim)
        
        self.conv_layers = nn.ModuleList([
            DecoderConvolutionalBlock(hidden_dim, embed_dim, config['decoder_kernel_size'], self.dropout)
            for _ in range(config['num_decoder_layers'])
        ])

        self.hidden_to_output = nn.Linear(hidden_dim, config['vocab_size_tgt'])

    def forward(self, prev_output_tokens, encoder_out_dict):
        positions = torch.arange(prev_output_tokens.size(1), device=prev_output_tokens.device).unsqueeze(0)

        token_embed = self.token_embedding(prev_output_tokens)
        pos_embed = self.position_embedding(positions)
        x = token_embed + pos_embed
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        prev_word_embedding = x
        x = self.embed_to_hidden(x)
        x = x.transpose(1, 2)

        for i, conv_layer in enumerate(self.conv_layers):
            # 将编码器梯度按解码器层数缩放 (3.4节)
            if self.training:
                encoder_out_dict['encoder_output'].requires_grad_(True)
                encoder_out_dict['encoder_output'] = encoder_out_dict['encoder_output'] / len(self.conv_layers)

            x = conv_layer(x, prev_word_embedding, encoder_out_dict)

        x = x.transpose(1, 2)
        x = self.hidden_to_output(x)
        return x

class ConvS2S(nn.Module):
    """
    顶层模型，整合编码器和解码器
    """
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config['model'])
        self.decoder = Decoder(config['model'])

    def forward(self, src_tokens, prev_output_tokens):
        encoder_out = self.encoder(src_tokens)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out