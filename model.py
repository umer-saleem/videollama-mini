import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VideoEncoderMini(nn.Module):
    def __init__(self, C=128, num_heads=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj = nn.Linear(64, C)
        self.temporal_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=C,
                nhead=num_heads,
                dim_feedforward=4 * C,
                batch_first=True
            ),
            num_layers=1
        )

    def forward(self, frames):
        B, F, C, H, W = frames.shape
        frames = frames.view(B * F, C, H, W)
        feats = self.cnn(frames).view(B, F, -1)
        feats = self.proj(feats)
        return self.temporal_attn(feats)


class PositionalEncoding(nn.Module):
    def __init__(self, C, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, C)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, C, 2).float() * (-math.log(10000.0) / C))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class QFormerBlockMini(nn.Module):
    def __init__(self, C=128):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(C, 4, batch_first=True)
        self.ln1 = nn.LayerNorm(C)
        self.ff = nn.Sequential(
            nn.Linear(C, 4 * C),
            nn.GELU(),
            nn.Linear(4 * C, C)
        )
        self.ln2 = nn.LayerNorm(C)

    def forward(self, text_tokens, visual_tokens):
        attn, _ = self.cross_attn(text_tokens, visual_tokens, visual_tokens)
        x = self.ln1(text_tokens + attn)
        return self.ln2(x + self.ff(x))


class VideoLLaMAMini(nn.Module):
    def __init__(self, vocab_size, C=128, n_layers=2, max_len=50):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, C)
        self.pos_emb = PositionalEncoding(C, max_len)
        self.video_encoder = VideoEncoderMini(C)
        self.qformer = QFormerBlockMini(C)

        self.decoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=C, nhead=4, batch_first=True)
            for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(C)
        self.head = nn.Linear(C, vocab_size)

    def forward(self, input_ids, video_frames):
        B, T = input_ids.shape
        x = self.pos_emb(self.token_emb(input_ids))
        v = self.video_encoder(video_frames)
        x = self.qformer(x, v)

        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device) * float("-inf"),
            diagonal=1
        )

        for layer in self.decoder:
            x = layer(x, src_mask=causal_mask)

        return self.head(self.ln(x))
