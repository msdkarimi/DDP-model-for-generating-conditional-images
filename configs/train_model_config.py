from ml_collections import ConfigDict
__all__ = ['get_config', 'get_config_trainer']

def get_config_trainer():
    trainer_configs = ConfigDict()
    trainer_configs.name = 'trainer'
    trainer_configs.logger_name = 'ddpm_trainer'
    trainer_configs.optimizer_config = _get_optimizer_config()
    return trainer_configs


def get_config():
    all_configs = ConfigDict()
    all_configs.model = _get_model_config()
    all_configs.optimizer = _get_optimizer_config()
    all_configs.lr_scheduler = _get_lr_scheduler_config()
    all_configs.ddpm = _get_ddpm_config()
    all_configs.dataloader = _get_dataloader_config()
    return all_configs

def _get_model_config():
    model_configs = ConfigDict()
    model_configs.module = 'open_ai_unet'
    model_configs.name = 'UNetModel'
    # TODO add model configs
    return model_configs

def _get_optimizer_config():
    opt_configs = ConfigDict()
    opt_configs.name = 'Adam'
    opt_configs.base_lr = 5e-4
    # TODO add opt_configs
    return opt_configs

def _get_lr_scheduler_config():
    lr_scheduler_configs = ConfigDict()
    # TODO add lr_scheduler_configs
    return lr_scheduler_configs

def _get_ddpm_config():
    ddpm_configs = ConfigDict()
    # TODO add ddpm_configs
    return ddpm_configs

def _get_dataloader_config():
    dataloader_configs = ConfigDict()
    dataloader_configs.batch_size = 4
    dataloader_configs.num_workers = 0
    # TODO add dataloader_configs
    return dataloader_configs

if __name__ == '__main__':
    print(get_config())
    exit(0)