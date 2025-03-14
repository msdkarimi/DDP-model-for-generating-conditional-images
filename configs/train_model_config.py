from ml_collections import ConfigDict
# __all__ = ['get_all_config', 'get_config_trainer']

def get_logger_config():
    logger_configs = ConfigDict()
    logger_configs.logger_name = 'ddpm_trainer_logger'
    return logger_configs


def get_config_trainer():
    trainer_configs = ConfigDict()
    trainer_configs.logger_name = 'ddpm_trainer'
    trainer_configs.mix_precision = False
    trainer_configs.optimizer_name = 'ADAM'
    trainer_configs.optimizer_base_lr = 5e-4
    trainer_configs.optimizer_weight_decay = -1 # TODO update
    trainer_configs.optimizer_betas = -1 # TODO update
    trainer_configs.n_epochs = 200
    return trainer_configs





def get_optimizer_config():
    opt_configs = ConfigDict()
    opt_configs.name = 'Adam'
    opt_configs.base_lr = 5e-4
    # TODO add opt_configs
    return opt_configs


def get_lr_scheduler_config():
    lr_scheduler_configs = ConfigDict()
    lr_scheduler_configs.name = 'linear'
    lr_scheduler_configs.t_in_epochs = False    # in order to update by batch not step
    lr_scheduler_configs.warmup_epochs = 0    # in order to update by batch not step
    lr_scheduler_configs.warmup_lr_init = 1e-6  # initial LR for warmup
    lr_scheduler_configs.lr_min_rate = .01  # to multiply to decay the LR
    return lr_scheduler_configs


def get_dataloader_config():
    dataloader_configs = ConfigDict()
    dataloader_configs.batch_size = 1
    dataloader_configs.num_workers = 0
    dataloader_configs.image_size = 256
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
    # diffusion_configs.image_size = 64
    diffusion_configs.image_size = 32
    # diffusion_configs.channels = 3
    diffusion_configs.channels = 4
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

# def get_unet_config():
#     """
#     this is for the 860M unet model
#     """
#     unet_configs = ConfigDict()
#     unet_configs.name = "open_ai_model"
#     unet_configs.image_size = 32
#     unet_configs.in_channels = 4
#     unet_configs.out_channels = 4
#     unet_configs.model_channels = 320
#     unet_configs.attention_resolutions = [4, 2, 1]
#     unet_configs.num_res_blocks = 2
#     unet_configs.channel_mult = [1, 2, 4, 4]
#     unet_configs.num_heads = 8
#     unet_configs.use_spatial_transformer = True
#     unet_configs.transformer_depth = 1
#     unet_configs.context_dim = 768 # 640
#     unet_configs.legacy = False
#     return unet_configs

def get_unet_config():
    """
    for model as big as model for CELEB_A
    """
    unet_configs = ConfigDict()
    unet_configs.name = "open_ai_model"
    unet_configs.image_size = 32
    unet_configs.in_channels = 4
    unet_configs.out_channels = 4
    unet_configs.model_channels = 192 #224
    unet_configs.attention_resolutions = [8, 4, 2]
    unet_configs.num_res_blocks = 2
    unet_configs.channel_mult = [1, 2, 3, 5]
    unet_configs.num_head_channels = 32
    # unet_configs.num_heads = 8 # instead of above
    unet_configs.use_spatial_transformer = True
    unet_configs.transformer_depth = 1
    unet_configs.context_dim = 768 # 640
    return unet_configs

def get_latent_diffusion_config():
    latent_diffusion_configs = ConfigDict()
    latent_diffusion_configs.name = 'latent_diffusion'
    latent_diffusion_configs.num_timesteps_cond = 1
    latent_diffusion_configs.cond_stage_key = "caption"
    latent_diffusion_configs.cond_stage_trainable = False
    # latent_diffusion_configs.concat_mode = True
    latent_diffusion_configs.cond_stage_forward = None
    latent_diffusion_configs.conditioning_key = 'crossattn'
    # latent_diffusion_configs.scale_factor = 1.0
    # latent_diffusion_configs.scale_by_std = False
    latent_diffusion_configs.scale_factor = 0.18215
    latent_diffusion_configs.scale_by_std = True


    return latent_diffusion_configs

def get_first_stage_config():
    first_stage_configs = ConfigDict()
    first_stage_configs.name = 'autoencoder'
    first_stage_configs.embed_dim = 4
    first_stage_configs.ckpt_path = "pretrained/vae_f_8.ckpt"  # None
    # first_stage_configs.ckpt_path = "../pretrained/vae_f_8.ckpt"  # None
    # first_stage_configs.monitor: val / rec_loss
    first_stage_configs.ddconfig = ConfigDict()
    first_stage_configs.ddconfig.double_z = True
    first_stage_configs.ddconfig.z_channels = 4
    first_stage_configs.ddconfig.resolution = 256
    first_stage_configs.ddconfig.in_channels = 3
    first_stage_configs.ddconfig.out_ch = 3
    first_stage_configs.ddconfig.ch = 128
    first_stage_configs.ddconfig.ch_mult = [1, 2, 4, 4]
    first_stage_configs.ddconfig.num_res_blocks = 2
    first_stage_configs.ddconfig.attn_resolutions = []
    first_stage_configs.ddconfig.dropout = 0.0
    return first_stage_configs


def get_conditioning_config():
    conditioning_configs = ConfigDict()
    conditioning_configs.name = 'clip_text_encoder'
    # conditioning_configs.version = "openai/clip-vit-large-patch14"
    conditioning_configs.version = "pretrained/clip_model"
    # conditioning_configs.n_embed = 768
    # conditioning_configs.n_layer = 32
    return conditioning_configs

def get_image_logger_config():
    image_logger_configs = ConfigDict()
    image_logger_configs.frequency = 1000 # freq. for logging image generation/diffusion
    image_logger_configs.rescale = True
    image_logger_configs.log_on = 'step'
    image_logger_configs.clamp = True
    return image_logger_configs

# kwargs = plot_denoise_rows = True/ -False, ddim_steps=200, plot_progressive_rows=True, log_every_t(in this way this could be completely different from the train one.)


def get_log_image_kwargs():
    image_logger_configs = ConfigDict()
    image_logger_configs.log_every_t = 25 # freq. for generation/diffusion step
    image_logger_configs.plot_progressive_rows = True
    image_logger_configs.ddim_steps = None
    image_logger_configs.plot_denoise_rows = False
    return image_logger_configs


def get_all_config():
    return get_unet_config(), get_diffusion_config(), get_first_stage_config(), get_conditioning_config(), get_latent_diffusion_config(), get_image_logger_config(), get_log_image_kwargs(), get_lr_scheduler_config()

if __name__ == '__main__':
    print(get_all_config())
    exit(0)