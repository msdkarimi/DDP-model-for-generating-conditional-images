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
                 logger_name='logger',
                 mix_precision=False,

                 **kwargs
                 ):

        self.logger = build_logger(logger_name)
        model_name = latent_diffusion_config.name
        del latent_diffusion_config.name
        self.l_d_model = builder(model_name,*(unet_configs, gaussian_diff_config, vae_config, text_embeder_config), **latent_diffusion_config)
        self.optimizer = None # TODO optimizer to do task
        self.lr_scheduler = None # TODO lr_scheduler
        self.fp_16 = mix_precision
        self.num_steps_per_epoch = -1 # TODO must be len (data_loader train)

    def train_one_step(self, epoch, batch_idx, batch):
        self.forward_backward_step(epoch, batch_idx, batch)

    def forward_backward_step(self, epoch, batch_idx, batch):
        loss, loss_dict = self.feed_forward(batch)
        if not self.fp_16:
            self.backpropagation(epoch, batch_idx, loss)
        else:
            # TODO implement the mix-precision
            raise NotImplementedError

    def feed_forward(self, batch)-> torch.Tensor:
        return self.l_d_model(batch)

    def backpropagation(self, epoch, batch_idx, loss:torch.Tensor):
        self.optimizer.zero_grad()
        # TODO implement the backpropagation
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step_update((epoch * self.num_steps_per_epoch) + batch_idx)

    def run_trainer(self, data_loader):
        for epoch in range(0, self.l_d_model.n_epochs):
            for idx, a_batch in enumerate(data_loader):
                # {'image': images, 'caption': captions}
                self.train_one_step(epoch, idx, a_batch)


@register_model
def build_trainer(*args, **configs):
    return Trainer(*args, **configs)
