import os
import torch
from deeparch.utils import set_seed, load_json
from deeparch.dataset import get_dataloaders
from deeparch.model_builder import build_model
from deeparch.train_utils import train_one_epoch, validate
from deeparch.utils import save_json, count_parameters
from deeparch.config import FULL_EPOCHS, OUTPUT_DIR

def train_full(config_path, output_dir=OUTPUT_DIR, epochs=FULL_EPOCHS):
    set_seed()
    cfg = load_json(config_path)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders()
    model = build_model(cfg).to(device)

    print("Model params:", count_parameters(model))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=cfg.get('weight_decay', 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val = 0.0
    history = {'train_acc':[], 'val_acc':[]}

    for ep in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, device)
        scheduler.step()
        history['train_acc'].append(train_acc); history['val_acc'].append(val_acc)
        print(f"Epoch {ep+1}/{epochs} - train_acc: {train_acc:.4f} - val_acc: {val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save({'model_state': model.state_dict(), 'config': cfg}, os.path.join(output_dir, "best_model.pth"))
    save_json(history, os.path.join(output_dir, "train_history.json"))
    print("Finished. Best val:", best_val)

if __name__ == "__main__":
    train_full(os.path.join(OUTPUT_DIR, "best_params.json"))