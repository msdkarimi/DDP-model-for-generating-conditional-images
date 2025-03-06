import torch
from utils.build import builder, register_model
from modules.logger import build_logger
import modules.latent_diffusion
from latent_ddpm_trainer.lr_scheduler import build_scheduler


class Trainer(object):
    def __init__(self,
                 unet_configs,
                 gaussian_diff_config,
                 vae_config,
                 text_embeder_config,
                 latent_diffusion_config,
                 lr_scheduler_config,
                 train_data_loader,
                 val_data_loader,

                 logger_name='logger',
                 mix_precision=False,
                 optimizer_name='',
                 optimizer_base_lr=1e-4,
                 optimizer_weight_decay=-1,
                 optimizer_betas=(.99, 0.99),
                 n_epochs=1,

                 **kwargs
                 ):

        self.num_steps_per_epoch = len(train_data_loader)
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

        self.logger = build_logger(logger_name)
        model_name = latent_diffusion_config.name
        del latent_diffusion_config.name
        self.l_d_model = builder(model_name, unet_configs, gaussian_diff_config, vae_config, text_embeder_config,
                                 **latent_diffusion_config)
        self.optimizer = None # TODO optimizer to do task
        self.lr_scheduler = build_scheduler(lr_scheduler_config, self.optimizer, self.num_steps_per_epoch, n_epochs)
        self.fp_16 = mix_precision

    def train_one_step(self, epoch, batch_idx, batch):
        self.forward_backward_step(epoch, batch_idx, batch)

    def forward_backward_step(self, epoch, batch_idx, batch):
        loss, loss_dict = self.feed_forward(batch)
        if not self.fp_16:

            self.backpropagation(epoch, batch_idx, loss)
        else:
            # TODO implement the mix-precision
            # TODO needs to handle the case that mp. returns false
            raise NotImplementedError

    def feed_forward(self, batch)-> torch.Tensor:
        return self.l_d_model(batch)

    def backpropagation(self, epoch, batch_idx, loss:torch.Tensor):
        self.optimizer.zero_grad()
        # TODO implement the backpropagation
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step_update((epoch * self.num_steps_per_epoch) + batch_idx)

    def run(self):

        for epoch in range(0, self.l_d_model.n_epochs):
            for idx, a_batch in enumerate(self.train_data_loader):
                # {'image': images, 'caption': captions}
                self.train_one_step(epoch, idx, a_batch)


@register_model
def build_trainer(*args, **configs):
    return Trainer(*args, **configs)
