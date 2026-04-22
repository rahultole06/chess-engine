import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=14, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(128 * 8 * 8, 1024)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))

        x = x.view(-1, 128 * 8 * 8)

        x = self.fc1(x)

        return x

class Engine(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, max_seq_len):
        super().__init__()

        self.cnn = ConvNet()

        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=d_model * 4,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        self.lm_head = nn.Linear(d_model, 4096)

    def forward(self, board_sequence):
        B, S_L, C, H, W = board_sequence.size()

        board_flatten = board_sequence.view(B * S_L, C, H, W)
        cnn_pass = self.cnn(board_flatten)

        x = cnn_pass.view(B, S_L, -1)
        
        positions = torch.arange(0, S_L, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(S_L).to(x.device)

        transformer_pass = self.transformer(x, mask=causal_mask, is_causal=True)

        logits = self.lm_head(transformer_pass)

        return logits