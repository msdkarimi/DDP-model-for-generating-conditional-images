import torch
from utils.build import builder, register_model
from modules.logger import build_logger
from modules.latent_diffusion import LatentDiffusion
from latent_ddpm_trainer.lr_scheduler import build_scheduler
from utils.base import LogHelper
from modules.image_logger import build_image_logger
import traceback
from utils.utils import log_dict, AverageMeter, get_grad_norm
from tqdm import tqdm


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
                 validate_every=5000, # this is for validation
                 log_every=50, # this is for logger
                 **kwargs
                 ):

        self.num_steps_per_epoch = len(train_data_loader)
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
        self.image_logger = build_image_logger(logger_name, logger_folder, log_image_kwargs, **image_logger_config)
        self.fp_16 = mix_precision
        self.validate_every = validate_every # TODO handle this in config
        self.log_every = log_every # TODO handle this in config
        self.loss_meter_simple = AverageMeter()
        self.loss_meter_vlb = AverageMeter()
        self.loss_meter_simple_ema = AverageMeter()
        self.grad_norm_meter = AverageMeter()

    def train_one_step(self, epoch, batch_idx, batch):
        self.l_d_model.model.train()
        self.forward_backward_step(epoch, batch_idx, batch)

    def forward_backward_step(self, epoch, batch_idx, batch):
        self.image_logger.do_log(self.l_d_model, self.num_steps_per_epoch, 'train', epoch, batch_idx+1, batch, )
        loss, loss_dict = self.feed_forward(batch)
        self.loss_meter_simple.update(loss_dict['train/loss_simple'].item())
        if (epoch*self.num_steps_per_epoch+batch_idx+1) % self.log_every == 0:
            loss_dict.update({'grad_norm': self.grad_norm_meter.avg})
            log_dict(loss_dict, self.logger, batch_idx, len(self.train_data_loader), epoch, self.loss_meter_simple.avg)
        self.backpropagation(epoch, batch_idx, loss)

    def feed_forward(self, batch)-> torch.Tensor:
        return self.l_d_model(batch)

    def backpropagation(self, epoch, batch_idx, loss:torch.Tensor):
        _step = (epoch * self.num_steps_per_epoch) + batch_idx
        if not self.fp_16:
            self.optimizer.zero_grad()
            loss.backward()
            self.grad_norm_meter.update(get_grad_norm(list(self.l_d_model.model.parameters())))
            self.optimizer.step()
            self.lr_scheduler.step_update(_step)
        else:
            # TODO needs to handle the case that mp. returns false
            raise NotImplementedError('fp16 is not implemented yet')

    def validate_one_batch(self, batch):
        # loss_dict_no_ema, loss_dict_ema = self.l_d_model.validation_step(batch)
        return self.l_d_model.validation_step(batch)

    @torch.no_grad()
    def validation(self, epoch, batch_idx):
        self.l_d_model.model.eval()

        # val_iterator = tqdm(enumerate(self.train_data_loader),
        #                 total=len(self.train_data_loader),
        #                 desc=f"Validating",
        #                 unit="batch")

        for idx, batch in enumerate(self.val_data_loader):
            # val_iterator.set_description(f"Validation/Batch [{idx}]")
            loss_dict_no_ema, loss_dict_ema = self.validate_one_batch(batch)
            if (idx + 1) % self.log_every == 0:
                log_dict(loss_dict_ema, self.logger, idx, len(self.val_data_loader))
            self.image_logger.do_log(self.l_d_model, self.num_steps_per_epoch, 'validation', epoch, batch_idx, batch)

    def run(self):
        # if self.check_pipeline():
        self.init_train()

    @torch.no_grad()
    def check_pipeline(self):
        try:
            self.logger.info('start checking pipeline')
            _a_batch = iter(self.train_data_loader).__next__()
            loss, loss_dict = self.feed_forward(_a_batch)
            log_dict(loss_dict, self.logger)
            self.image_logger.do_log(self.l_d_model, self.num_steps_per_epoch, 'validation', -1, 1, _a_batch)
            loss_dict_no_ema, loss_dict_ema = self.validate_one_batch(_a_batch)
            log_dict(loss_dict_ema, self.logger)
            self.logger.info('finish checking pipeline')
            return True
        except Exception as e:
            self.logger.error(e)
            self.logger.error(traceback.format_exc())
            return False

    def init_train(self):
        for epoch in range(0, self.n_epochs):
            # iterator = tqdm(enumerate(self.train_data_loader),
            #                 total=len(self.train_data_loader),
            #                 desc=f"Train/Epoch [{epoch + 1}]",
            #                 unit="batch")

            # for batch_idx, a_batch in enumerate(self.train_data_loader):
            for batch_idx, a_batch in enumerate(self.train_data_loader):
                # iterator.set_description(f"Train/Epoch [{epoch + 1}] >> Batch [{batch_idx + 1}]")
                self.train_one_step(epoch, batch_idx, a_batch)
                if (epoch * batch_idx + batch_idx+1) % self.validate_every == 0:
                    self.validation(epoch, batch_idx)

@register_model
def build_trainer(*args, **configs):
    return Trainer(*args, **configs)
