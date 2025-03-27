import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        Returns:
            x with positional encodings added.
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class BiteSenseModel(nn.Module):
    """
    Transformer-based model that performs hierarchical classification and parallel regression.
    
    Expected input shape: (batch_size, seq_len, input_dim)
    Hierarchical outputs:
      - Food state, texture, nutritional value, cooking method, food type.
    Regression outputs:
      - Bite count and eating duration.
    """
    def __init__(self, input_dim, embed_dim=128, num_layers=4, num_heads=8, ff_dim=256, dropout=0.1,
                 num_classes_state=3, num_classes_texture=2, num_classes_nutritional=3,
                 num_classes_cooking=2, num_classes_food=10):
        super(BiteSenseModel, self).__init__()
        # Project raw input features to the embedding space.
        self.input_linear = nn.Linear(input_dim, embed_dim)
        # Learnable class token for sequence-level representation.
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=ff_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Hierarchical classification heads:
        self.fc_state = nn.Linear(embed_dim, num_classes_state)
        self.fc_texture = nn.Linear(embed_dim, num_classes_texture)
        self.fc_nutritional = nn.Linear(embed_dim, num_classes_nutritional)
        self.fc_cooking = nn.Linear(embed_dim, num_classes_cooking)
        self.fc_food = nn.Linear(embed_dim, num_classes_food)
        
        # Parallel regression heads:
        self.fc_bite_count = nn.Linear(embed_dim, 1)
        self.fc_duration = nn.Linear(embed_dim, 1)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            A dictionary with classification logits and regression outputs.
        """
        batch_size = x.size(0)
        x = self.input_linear(x)  # (batch_size, seq_len, embed_dim)
        
        # Prepend the class token to each instance.
        class_token = self.class_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((class_token, x), dim=1)  # (batch_size, seq_len+1, embed_dim)
        
        # Transformer expects input shape: (seq_len+1, batch_size, embed_dim)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # Use the class token output as the summary representation.
        cls_out = x[0]  # (batch_size, embed_dim)
        
        # Hierarchical classification outputs.
        out_state = self.fc_state(cls_out)
        out_texture = self.fc_texture(cls_out)
        out_nutritional = self.fc_nutritional(cls_out)
        out_cooking = self.fc_cooking(cls_out)
        out_food = self.fc_food(cls_out)
        
        # Parallel regression outputs.
        out_bite_count = self.fc_bite_count(cls_out)
        out_duration = self.fc_duration(cls_out)
        
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
    input_dim = 90  # This should match the dimension of the feature vector extracted from CSV.
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    
    model = BiteSenseModel(input_dim=input_dim, embed_dim=128, num_layers=4, num_heads=8,
                           ff_dim=256, dropout=0.1,
                           num_classes_state=3, num_classes_texture=2, num_classes_nutritional=3,
                           num_classes_cooking=2, num_classes_food=10)
    outputs = model(dummy_input)
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")
