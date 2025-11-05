def sample_architecture(trial):
    cfg = {}
    cfg['n_conv_blocks'] = trial.suggest_int('n_conv_blocks', 2, 4)   # small for prototyping
    cfg['activation'] = trial.suggest_categorical('activation', ['relu', 'leakyrelu'])
    cfg['n_fc'] = trial.suggest_int('n_fc', 0, 1)
    cfg['fc_use_bn'] = trial.suggest_categorical('fc_use_bn', [True, False])
    cfg['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)

    for i in range(cfg['n_conv_blocks']):
        cfg[f'filters_{i}'] = trial.suggest_categorical(f'filters_{i}', [32, 64, 128])
        cfg[f'kernel_{i}'] = trial.suggest_categorical(f'kernel_{i}', [3, 5])
        cfg[f'use_bn_{i}'] = trial.suggest_categorical(f'use_bn_{i}', [True, False])
        cfg[f'pool_{i}'] = trial.suggest_categorical(f'pool_{i}', ['max', 'avg'])

    for j in range(cfg['n_fc']):
        cfg[f'fc_units_{j}'] = trial.suggest_categorical(f'fc_units_{j}', [128, 256])

    cfg['optimizer'] = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    cfg['lr'] = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    cfg['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    return cfg