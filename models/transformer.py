import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :].expand(-1, x.size(1), -1)
        return self.dropout(x)

class BiteSenseModel(nn.Module):
    def __init__(self, input_dims, embed_dim=128, num_layers=2, num_heads=4, ff_dim=256, dropout=0.1,
                 num_classes_state=3, num_classes_texture=2, num_classes_nutritional=3,
                 num_classes_cooking=2, num_classes_food=10):
        super(BiteSenseModel, self).__init__()

        self.num_sets = len(input_dims)  # f1 to f5

        # Separate input projectors and encoders per feature set
        self.input_linears = nn.ModuleList([
            nn.Linear(in_dim, embed_dim) for in_dim in input_dims
        ])

        self.pos_encoders = nn.ModuleList([
            PositionalEncoding(embed_dim, dropout) for _ in range(self.num_sets)
        ])

        self.class_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, embed_dim)) for _ in range(self.num_sets)
        ])

        self.encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                            dim_feedforward=ff_dim, dropout=dropout),
                num_layers=num_layers
            ) for _ in range(self.num_sets)
        ])

        # Total concatenated latent dim after all encoders
        total_latent_dim = embed_dim * self.num_sets

        # Classification heads
        self.fc_state = nn.Linear(total_latent_dim, num_classes_state)
        self.fc_texture = nn.Linear(total_latent_dim, num_classes_texture)
        self.fc_nutritional = nn.Linear(total_latent_dim, num_classes_nutritional)
        self.fc_cooking = nn.Linear(total_latent_dim, num_classes_cooking)
        self.fc_food = nn.Linear(total_latent_dim, num_classes_food)

        # Regression heads
        self.fc_bite_count = nn.Linear(total_latent_dim, 1)
        self.fc_duration = nn.Linear(total_latent_dim, 1)

    def forward(self, inputs):
        # inputs: list of 5 tensors, each of shape (batch, seq_len, input_dim_i)
        latent_outputs = []
        for i in range(self.num_sets):
            x = self.input_linears[i](inputs[i])  # (batch, seq, embed_dim)
            bsz = x.size(0)
            cls_token = self.class_tokens[i].expand(bsz, -1, -1)  # (batch, 1, embed_dim)
            x = torch.cat((cls_token, x), dim=1)  # (batch, seq+1, embed_dim)
            x = x.transpose(0, 1)  # (seq+1, batch, embed_dim)
            x = self.pos_encoders[i](x)
            x = self.encoders[i](x)  # (seq+1, batch, embed_dim)
            x = x.transpose(0, 1)  # (batch, seq+1, embed_dim)
            latent_outputs.append(x)

        # Concat latent features across feature sets
        concat_latents = torch.cat(latent_outputs, dim=-1)  # (batch, seq+1, total_latent_dim)

        # Pooling (e.g., mean across sequence)
        pooled = concat_latents.mean(dim=1)  # (batch, total_latent_dim)

        # Heads
        out_state = self.fc_state(pooled)
        out_texture = self.fc_texture(pooled)
        out_nutritional = self.fc_nutritional(pooled)
        out_cooking = self.fc_cooking(pooled)
        out_food = self.fc_food(pooled)

        out_bite_count = self.fc_bite_count(pooled)
        out_duration = self.fc_duration(pooled)

        return {
            "state": out_state,
            "texture": out_texture,
            "nutritional": out_nutritional,
            "cooking": out_cooking,
            "food": out_food,
            "bite_count": out_bite_count,
            "duration": out_duration
        }

if __name__ == "__main__":
    # Test the model with dummy input.
    batch_size = 16
    seq_len = 50
    input_dims = [102, 30, 20, 15, 25]  # Example feature dimensions for f1 to f5
    dummy_inputs = [torch.randn(batch_size, seq_len, dim) for dim in input_dims]

    model = BiteSenseModel(input_dims=input_dims, embed_dim=128, num_layers=4, num_heads=8,
                           ff_dim=256, dropout=0.1,
                           num_classes_state=3, num_classes_texture=2, num_classes_nutritional=3,
                           num_classes_cooking=2, num_classes_food=10)
    outputs = model(dummy_inputs)
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")
