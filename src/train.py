import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import BraTSSliceDataset
from model_unet import get_unet
from utils import bce_dice_loss, dice_coeff

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

def collate_fn(batch):
    
    batch = [b for b in batch if b is not None]
    return torch.utils.data.default_collate(batch) if len(batch) > 0 else None

def run_training(case_folders, out_dir='checkpoints', epochs=30, batch_size=8, lr=1e-4, device=device):
    os.makedirs(out_dir, exist_ok=True)

    print("Initializing dataset...")
    train_ds = BraTSSliceDataset(case_folders, with_mask=True)
    print(f"Dataset ready: {len(train_ds)} samples found")

    print("Preparing DataLoader...")
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    print(f"DataLoader initialized | Batch size: {batch_size} | Num workers: 4")
    
    print("Initializing model and optimizer...")
    model = get_unet().to(device)
    print(f"Model loaded to device: {device} | Learning rate: {lr}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"Starting training for {epochs} epochs...")
    best_dice = 0.0
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(tr_loader, desc=f"Epoch {epoch}"):
            imgs = batch['image'].to(device)  # [B,4,H,W]
            masks = batch['mask'].to(device)  # [B,1,H,W]
            preds = model(imgs)
            loss = bce_dice_loss(preds, masks)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(tr_loader)
        
        model.eval()
        with torch.no_grad():
            imgs = next(iter(tr_loader))['image'].to(device)
            masks = next(iter(tr_loader))['mask'].to(device)
            preds = model(imgs)
            d = dice_coeff(preds, masks)
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, sample_val_dice={d:.4f}")
        if d > best_dice:
            best_dice = d
            torch.save(model.state_dict(), os.path.join(out_dir, 'best.pt'))
    print("Training finished. Best dice:", best_dice)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_dir", type=str, required=True, help="directory containing case subfolders")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--out", type=str, default="checkpoints")
    args = parser.parse_args()
    case_dirs = [os.path.join(args.cases_dir, d) for d in os.listdir(args.cases_dir) if os.path.isdir(os.path.join(args.cases_dir, d))]
    run_training(case_dirs, out_dir=args.out, epochs=args.epochs, batch_size=args.batch)