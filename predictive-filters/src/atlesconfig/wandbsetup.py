from src.atlesconfig import config

def set_wandb_config(wandb):
    # wandb.init should be called before calling this function.
    wandb.config.learning_rate = config.get_config(section='ml', key='lr')
    wandb.config.weight_decay = config.get_config(section='ml', key='weight_decay')
    wandb.config.embedding_dim = config.get_config(section='ml', key='embedding_dim')
    wandb.config.encoder_layers = config.get_config(section='ml', key='encoder_layers')
    wandb.config.num_heads = config.get_config(section='ml', key='num_heads')
    wandb.config.dropout = config.get_config(section='ml', key='dropout')
    wandb.config.mse_weight = config.get_config(section='ml', key='mse_weight')
    wandb.config.ce_weight_clv = config.get_config(section='ml', key='ce_weight_clv')
    wandb.config.ce_weight_mod = config.get_config(section='ml', key='ce_weight_mod')