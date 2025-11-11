import os
import optuna
from deeparch.objective import objective
from deeparch.utils import set_seed, save_json
from deeparch.config import N_TRIALS, OUTPUT_DIR
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

def run_search(n_trials=N_TRIALS, output_dir=OUTPUT_DIR):
    set_seed()
    os.makedirs(output_dir, exist_ok=True)
    sampler = TPESampler()
    pruner = MedianPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_trial.params
    
    save_json(best_params, os.path.join(output_dir, "best_params.json"))
    print("Best val acc:", study.best_value)
    print("Best params saved to", os.path.join(output_dir, "best_params.json"))

if __name__ == "__main__":
    run_search()