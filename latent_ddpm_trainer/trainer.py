import torch
from utils.build import builder, register_model
from modules.logger import build_logger
import modules.latent_diffusion





class Trainer(object):
    def __init__(self,
                 unet_configs,
                 gaussian_diff_config,
                 vae_config,
                 text_embeder_config,
                 latent_diffusion_config,
                 logger_name = 'logger',
                 **kwargs
                 ):

        self.logger = build_logger(logger_name)
        model_name = latent_diffusion_config.name
        del latent_diffusion_config.name
        self.ldm_model = builder(model_name,*(unet_configs, gaussian_diff_config, vae_config, text_embeder_config), **latent_diffusion_config)


    def train_step(self, images, labels=None):
        self.forward_backward_step(images, labels)

    def forward_backward_step(self, images, labels):
        loss = self.feed_forward(images, labels)
        self.backpropagation(loss)


    @staticmethod
    def feed_forward(images, labels)-> torch.Tensor:
        return 0

    @staticmethod
    def backpropagation(loss:torch.Tensor):
        pass

@register_model
def build_trainer(*args, **configs):
    return Trainer(*args, **configs)
