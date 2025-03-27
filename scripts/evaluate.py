import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from transformer import BiteSenseModel
import utils

class IMUEvalDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, fs=44.0, window_duration=3.0, step_duration=1.5):
        self.raw_data = utils.load_imu_csv(csv_path)
        self.features = utils.process_imu_data(self.raw_data, fs=fs,
                                               window_duration=window_duration,
                                               step_duration=step_duration)
        self.feature_dim = self.features.shape[1]
        self.seq_len = self.features.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        features = torch.tensor(self.features, dtype=torch.float32)
        return features

def evaluate(args):
    # Load evaluation data.
    dataset = IMUEvalDataset(csv_path=args.data_path, fs=args.fs,
                              window_duration=args.window_duration,
                              step_duration=args.step_duration)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Instantiate model and load checkpoint.
    model = BiteSenseModel(input_dim=dataset.feature_dim,
                           embed_dim=args.embed_dim,
                           num_layers=args.num_layers,
                           num_heads=args.num_heads,
                           ff_dim=args.ff_dim,
                           dropout=args.dropout,
                           num_classes_state=args.num_classes_state,
                           num_classes_texture=args.num_classes_texture,
                           num_classes_nutritional=args.num_classes_nutritional,
                           num_classes_cooking=args.num_classes_cooking,
                           num_classes_food=args.num_classes_food)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    with torch.no_grad():
        for features in dataloader:
            features = features.to(device)
            outputs = model(features)
            # For demonstration, we simply print the outputs.
            print("Model Outputs:")
            print("Food State Logits:", outputs["state"])
            print("Food Texture Logits:", outputs["texture"])
            print("Nutritional Value Logits:", outputs["nutritional"])
            print("Cooking Method Logits:", outputs["cooking"])
            print("Food Type Logits:", outputs["food"])
            print("Predicted Bite Count:", outputs["bite_count"])
            print("Predicted Eating Duration:", outputs["duration"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BiteSense Model")
    parser.add_argument("--data_path", type=str, default="data/raw/imu_data.csv",
                        help="Path to the CSV file containing raw IMU data for evaluation")
    parser.add_argument("--checkpoint", type=str, default="models/bitesense_model.pth",
                        help="Path to the saved model checkpoint")
    parser.add_argument("--fs", type=float, default=44.0, help="Sampling frequency")
    parser.add_argument("--window_duration", type=float, default=3.0, help="Window duration in seconds")
    parser.add_argument("--step_duration", type=float, default=1.5, help="Sliding window step in seconds")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension for the model")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--ff_dim", type=int, default=256, help="Feed-forward network dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--num_classes_state", type=int, default=3, help="Number of classes for food state")
    parser.add_argument("--num_classes_texture", type=int, default=2, help="Number of classes for food texture")
    parser.add_argument("--num_classes_nutritional", type=int, default=3, help="Number of classes for nutritional value")
    parser.add_argument("--num_classes_cooking", type=int, default=2, help="Number of classes for cooking method")
    parser.add_argument("--num_classes_food", type=int, default=10, help="Number of classes for food type")
    
    args = parser.parse_args()
    evaluate(args)
