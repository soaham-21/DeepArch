import torch
import optuna
from deeparch.search_space import sample_architecture
from deeparch.model_builder import build_model
from deeparch.dataset import get_dataloaders
from deeparch.train_utils import train_one_epoch, validate
from deeparch.config import PARTIAL_EPOCHS, BATCH_SIZE

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE, val_fraction=0.1)

    cfg = sample_architecture(trial)
    model = build_model(cfg).to(device)

    if cfg['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=cfg['weight_decay'])

    best_val = 0.0
    for epoch in range(PARTIAL_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, device)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        if val_acc > best_val:
            best_val = val_acc
    return best_val