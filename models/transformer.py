import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding.
    """
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
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
        """
        x: (sequence_length, batch_size, d_model)
        Returns: x augmented with positional encodings.
        """
        x = x + self.pe[:x.size(0), :].expand(-1, x.size(1), -1)
        return self.dropout(x)

class BiteSenseModel(nn.Module):
    """
    Transformer-based model for chewing analysis.
    
    This model processes multiple sets of features extracted from IMU data
    (each set representing a distinct aspect of the chewing process as described in the paper).
    
    For each feature set:
      - A linear projection maps the input to a common latent space.
      - A learnable class token is prepended to the sequence.
      - Sinusoidal positional encodings are added.
      - The sequence is processed by a dedicated Transformer encoder.
    
    The latent representations from all feature sets are concatenated along the feature dimension,
    then pooled before being passed to hierarchical classification and regression heads.
    """
    def __init__(self, input_dims, embed_dim=128, num_layers=2, num_heads=4, ff_dim=256, dropout=0.1,
                 num_classes_state=3, num_classes_texture=2, num_classes_nutritional=3,
                 num_classes_cooking=2, num_classes_food=10):
        """
        Args:
            input_dims (list of int): A list where each element is the dimension of a feature set.
            embed_dim: Dimension to which each feature set is projected.
            num_layers: Number of Transformer encoder layers per feature set.
            num_heads: Number of attention heads in each encoder layer.
            ff_dim: Dimension of the feedforward network within each encoder layer.
            dropout: Dropout probability.
            num_classes_*: Number of classes for each hierarchical classification task.
        """
        super().__init__()
        self.num_sets = len(input_dims)  # Number of feature sets
        
        # Separate modules for each feature set.
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
        
        # After processing, the latent representations are concatenated.
        total_latent_dim = embed_dim * self.num_sets
        
        # Hierarchical classification heads.
        self.fc_state = nn.Linear(total_latent_dim, num_classes_state)
        self.fc_texture = nn.Linear(total_latent_dim, num_classes_texture)
        self.fc_nutritional = nn.Linear(total_latent_dim, num_classes_nutritional)
        self.fc_cooking = nn.Linear(total_latent_dim, num_classes_cooking)
        self.fc_food = nn.Linear(total_latent_dim, num_classes_food)
        
        # Regression heads for continuous outputs.
        self.fc_bite_count = nn.Linear(total_latent_dim, 1)
        self.fc_duration = nn.Linear(total_latent_dim, 1)

    def forward(self, inputs):
        """
        Args:
            inputs: A list of tensors, one per feature set.
                    Each tensor should have shape (batch_size, sequence_length, feature_dimension),
                    where feature_dimension corresponds to the features extracted as described in the paper.
        Returns:
            A dictionary with outputs for each classification and regression head.
        """
        latent_outputs = []
        for i in range(self.num_sets):
            # Project the input feature set into the latent space.
            x = self.input_linears[i](inputs[i])
            batch_size = x.size(0)
            # Prepend the learnable class token.
            cls_token = self.class_tokens[i].expand(batch_size, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            # Transformer expects input as (sequence_length, batch_size, embed_dim)
            x = x.transpose(0, 1)
            x = self.pos_encoders[i](x)
            x = self.encoders[i](x)
            x = x.transpose(0, 1)  # Back to (batch_size, sequence_length, embed_dim)
            latent_outputs.append(x)
        
        # Concatenate latent features from all sets.
        concat_latents = torch.cat(latent_outputs, dim=-1)
        # Pool across the sequence dimension (e.g., mean pooling).
        pooled = concat_latents.mean(dim=1)
        
        # Classification heads.
        out_state = self.fc_state(pooled)
        out_texture = self.fc_texture(pooled)
        out_nutritional = self.fc_nutritional(pooled)
        out_cooking = self.fc_cooking(pooled)
        out_food = self.fc_food(pooled)
        
        # Regression heads.
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
    # Test the model with dummy inputs.
    batch_size = 16
    seq_len = 50
    input_dims = input_dims
    
    dummy_inputs = [torch.randn(batch_size, seq_len, dim) for dim in input_dims]
    
    model = BiteSenseModel(
        input_dims=input_dims,
        embed_dim=128,
        num_layers=2,
        num_heads=4,
        ff_dim=256,
        dropout=0.1,
        num_classes_state=3,
        num_classes_texture=2,
        num_classes_nutritional=3,
        num_classes_cooking=2,
        num_classes_food=10
    )
    
    outputs = model(dummy_inputs)
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")
