import torch
from utils.build import builder, register_model
from modules.logger import build_logger
from modules.latent_diffusion import LatentDiffusion
from latent_ddpm_trainer.lr_scheduler import build_scheduler
from utils.base import LogHelper
from modules.image_logger import build_image_logger
import traceback


class Trainer(LogHelper):
    def __init__(self,
                 unet_configs,
                 gaussian_diff_config,
                 vae_config,
                 text_embeder_config,
                 latent_diffusion_config,
                 image_logger_config,
                 log_image_kwargs,
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

        self.num_steps_per_epoch = 10 #len(train_data_loader)
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

        self.logger, logger_folder = build_logger(logger_name)
        l_d_model_name = latent_diffusion_config.name
        del latent_diffusion_config.name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.l_d_model:LatentDiffusion = builder(l_d_model_name, unet_configs, gaussian_diff_config, vae_config, text_embeder_config,
                                 **latent_diffusion_config).to(self.device)
        self.optimizer = torch.optim.AdamW(list(self.l_d_model.model.parameters()), lr=optimizer_base_lr) # TODO optimizer to do task
        self.lr_scheduler = build_scheduler(lr_scheduler_config, self.optimizer, self.num_steps_per_epoch, n_epochs)
        self.n_epochs = n_epochs
        # self.image_logger = build_image_logger(config_image_logger)
        self.image_logger = build_image_logger(logger_name, logger_folder, log_image_kwargs, image_logger_config)
        self.fp_16 = mix_precision

    def train_one_step(self, epoch, batch_idx, batch):
        self.l_d_model.model.train()
        self.forward_backward_step(epoch, batch_idx, batch)

    def forward_backward_step(self, epoch, batch_idx, batch):
        self.image_logger.do_log(self.l_d_model, self.num_steps_per_epoch, 'train', epoch, batch_idx+1, batch, )
        loss, loss_dict = self.feed_forward(batch)
        self.backpropagation(epoch, batch_idx, loss)

    def feed_forward(self, batch)-> torch.Tensor:
        return self.l_d_model(batch)


    def backpropagation(self, epoch, batch_idx, loss:torch.Tensor):
        _step = (epoch * self.num_steps_per_epoch) + batch_idx
        if self.fp_16:
            self.optimizer.zero_grad()
            # TODO implement the backpropagation
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step_update(_step)
        else:
            # TODO implement the mix-precision
            # TODO needs to handle the case that mp. returns false
            raise NotImplementedError

    def validate_one_batch(self, batch):
        # loss_dict_no_ema, loss_dict_ema = self.l_d_model.validation_step(batch)
        return self.l_d_model.validation_step(batch)



    @torch.no_grad()
    def validation(self, epoch, batch_idx):
        self.l_d_model.model.eval()
        for idx, batch in enumerate(self.val_data_loader):# TODO use tqdm
            loss_dict_no_ema, loss_dict_ema = self.validate_one_batch(batch)
            self.image_logger.do_log(self.l_d_model, self.num_steps_per_epoch, 'validation', epoch, batch_idx, batch)
            # TODO log the output

    def run(self):
        if self.check_pipeline():
            self.init_train()

    @torch.no_grad()
    def check_pipeline(self):
        try:
            self.logger.info('begin checking pipeline')
            _a_batch = iter(self.train_data_loader).__next__()
            self.logger.info('data loaded works!')
            loss, loss_dict = self.feed_forward(_a_batch)
            self.logger.info('feed forward of ldm works!')
            self.image_logger.do_log(self.l_d_model, self.num_steps_per_epoch, 'validation', -1, 1, _a_batch)
            self.logger.info('image logger of ldm works!')
            loss_dict_no_ema, loss_dict_ema = self.validate_one_batch(_a_batch)
            self.logger.info('ema model works!')
            return True
        except Exception as e:
            self.logger.error(e)
            self.logger.error(traceback.format_exc())
            return False

    def init_train(self):
        for epoch in range(0, self.n_epochs):
            for batch_idx, a_batch in enumerate(self.train_data_loader):# TODO use tqdm
                self.train_one_step(epoch, batch_idx, a_batch)
                if (epoch * batch_idx + batch_idx) % 1000 == 0: # TODO handle the log_every in config
                    self.validation(epoch, batch_idx)


@register_model
def build_trainer(*args, **configs):
    return Trainer(*args, **configs)
