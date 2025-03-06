from ml_collections import ConfigDict
# __all__ = ['get_all_config', 'get_config_trainer']

def get_logger_config():
    logger_configs = ConfigDict()
    logger_configs.logger_name = 'ddpm_trainer_logger'
    return logger_configs


def get_config_trainer():
    trainer_configs = ConfigDict()
    trainer_configs.name = 'latent_ddpm_trainer'
    trainer_configs.logger_name = 'ddpm_trainer'
    # trainer_configs.optimizer_config = get_optimizer_config()
    return trainer_configs, get_model_config()





def get_model_config():
    model_configs = ConfigDict()
    model_configs.module = 'open_ai_unet'
    model_configs.name = 'UNetModel'
    # TODO add model configs
    return model_configs

def get_optimizer_config():
    opt_configs = ConfigDict()
    opt_configs.name = 'Adam'
    opt_configs.base_lr = 5e-4
    # TODO add opt_configs
    return opt_configs

def get_lr_scheduler_config():
    lr_scheduler_configs = ConfigDict()
    # TODO add lr_scheduler_configs
    return lr_scheduler_configs

def get_ddpm_config():
    ddpm_configs = ConfigDict()
    # TODO add ddpm_configs
    return ddpm_configs

def get_dataloader_config():
    dataloader_configs = ConfigDict()
    dataloader_configs.batch_size = 1
    dataloader_configs.num_workers = 0
    dataloader_configs.image_size = 256
    # TODO add dataloader_configs
    return dataloader_configs

def get_diffusion_config():
    diffusion_configs = ConfigDict()
    # diffusion_configs.name = 'gaussian_diffusion'
    diffusion_configs.linear_start = 0.0015
    diffusion_configs.linear_end = 0.0195
    diffusion_configs.timesteps = 1000
    diffusion_configs.beta_schedule = 'linear'
    diffusion_configs.loss_type = 'l2'
    diffusion_configs.first_stage_key = 'image'
    # diffusion_configs.cond_stage_key = 'caption'
    diffusion_configs.image_size = 64
    diffusion_configs.channels = 3
    # diffusion_configs.conditioning_key = 'crossattn'
    diffusion_configs.monitor = 'val / loss_simple_ema'
    diffusion_configs.use_ema = True
    diffusion_configs.clip_denoised = True
    diffusion_configs.l_simple_weight = 1.
    diffusion_configs.use_positional_encodings = False
    diffusion_configs.learn_logvar = False
    diffusion_configs.logvar_init = 0.
    diffusion_configs.parameterization="eps"  # all assuming fixed variance schedules

    # diffusion_configs.unet_config = get_unet_config()

    return diffusion_configs

def get_unet_config():
    unet_configs = ConfigDict()
    unet_configs.name = "open_ai_model"
    unet_configs.image_size = 64
    unet_configs.in_channels = 3
    unet_configs.out_channels = 3
    unet_configs.model_channels = 192
    unet_configs.attention_resolutions = [8, 4, 2]
    unet_configs.num_res_blocks = 2
    unet_configs.channel_mult = [1, 2, 3, 5]
    unet_configs.num_head_channels = 32
    unet_configs.use_spatial_transformer = True
    unet_configs.transformer_depth = 1
    unet_configs.context_dim = 640
    return unet_configs

def get_latent_diffusion_config():
    latent_diffusion_configs = ConfigDict()
    latent_diffusion_configs.name = 'latent_diffusion'
    latent_diffusion_configs.num_timesteps_cond = 1
    latent_diffusion_configs.cond_stage_key = "caption"
    latent_diffusion_configs.cond_stage_trainable = True
    # latent_diffusion_configs.concat_mode = True
    latent_diffusion_configs.cond_stage_forward = None
    latent_diffusion_configs.conditioning_key = 'crossattn'
    latent_diffusion_configs.scale_factor = 1.0
    latent_diffusion_configs.scale_by_std = False


    return latent_diffusion_configs

def get_first_stage_config():
    first_stage_configs = ConfigDict()
    first_stage_configs.name = 'autoencoder'

    # TODO decide which one KL/VQ VAE - maybe KL!!!
    return first_stage_configs


def get_conditioning_config():
    conditioning_configs = ConfigDict()
    conditioning_configs.name = 'clip'
    conditioning_configs.n_embed = 640
    conditioning_configs.n_layer = 32
    return conditioning_configs


def get_all_config():
    return get_unet_config(), get_diffusion_config(), get_first_stage_config(), get_conditioning_config(), get_latent_diffusion_config()

if __name__ == '__main__':
    print(get_all_config())
    exit(0)