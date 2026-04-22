import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from engine import Engine
from dataset import Dataset

def train_engine(dataset_path, resume_checkpoint=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    d_model = 1024
    num_heads = 8
    num_layers = 6
    max_seq_len = 256
    batch_size = 16
    model = Engine(d_model, num_heads, num_layers, max_seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming training from {resume_checkpoint}...")

        checkpoint = torch.load(resume_checkpoint, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded model and optimizer states")
    else:
        print("Training fresh model")

    criterion = nn.CrossEntropyLoss()
    dataset = Dataset(dataset_path, max_seq_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    model.train()
    running_loss = 0.0
    
    print("Starting training phase...")
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)

        outputs_flat = outputs.view(-1, 4096)
        labels_flat = labels.view(-1)
        loss = criterion(outputs_flat, labels_flat)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 100 == 99:
            print(f"Batch {batch_idx + 1} | Loss: {running_loss / 100:.4f}")
            running_loss = 0.0
            
    print("Saving checkpoint...")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, "checkpoints/chess_net_supervised.pth")
    print("Supervised training complete!")

train_engine("dataset/lichess_db_standard_rated_2014-07.pgn.zst")