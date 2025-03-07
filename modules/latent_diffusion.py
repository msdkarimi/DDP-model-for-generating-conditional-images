from modules.gaussian_diffusion import GaussianDiffusion
from modules.clip_text_encoder  import FrozenCLIPEmbedder
import torch
from utils.diffusion_util import noise_like
from utils.ldm_util import default
from einops import rearrange, repeat
from utils.build import builder, register_model
from modules.distributions import DiagonalGaussianDistribution

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


class LatentDiffusion(GaussianDiffusion):
    def __init__(self, kwargs_unet_config, kwargs_diffusion_config, kwargs_autoencoder, kwargs_conditioning_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 conditioning_key=None,
                 cond_stage_forward=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 ):
        vae_name = kwargs_autoencoder.name
        del kwargs_autoencoder.name
        conditioner_name = kwargs_conditioning_config.name
        del kwargs_conditioning_config.name

        self.cond_stage_forward = cond_stage_forward
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs_diffusion_config['timesteps']
        super().__init__(kwargs_unet_config, **kwargs_diffusion_config)
        # side models
        self.first_stage_model = builder(vae_name, **kwargs_autoencoder)
        self.cond_stage_model:FrozenCLIPEmbedder = builder(conditioner_name, **kwargs_conditioning_config)

        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key

        try:
            self.num_downs = len(kwargs_autoencoder.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0

        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    def p_mean_variance(self, x, c, t,
                        clip_denoised: bool, quantize_denoised=False, corrector_kwargs=None):
        """
        based on models prediction of noise(aka eps), and given t, tries to predict the x_0, then computes the posterior of mu and var
        """

        model_output = self.model(x, t, c)

        if self.parameterization == 'eps':
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_output)
        else:
            raise NotImplementedError (f'Currently only supports {self.parameterization} parameterization')

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):


        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, c=c, t=t,
                                       clip_denoised=clip_denoised,
                                       quantize_denoised=quantize_denoised,
                                       corrector_kwargs=corrector_kwargs)

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise


    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):
        # TODO implement the sampler function
        pass


    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None,**kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            # condition must be list
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)


    def p_losses(self, x_start, cond, t, noise=None):
        # since we do not learn the variance thus the ELBO calculation is not feasible.
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'


        if self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError(f'only supports parameterization == {self.parameterization}')

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t

        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict


    def _forward_to_net(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None, 'conditioning key must be provided!'
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, t, *args, **kwargs)


    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        """
        this function return the z and raw condition which is text [caption]
        """
        x = super().get_input(batch, k)  # return jus the RGB image
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox']:
                    xc = batch[cond_key] # this
                    c = xc  # this wil be applied
                    if bs is not None:
                        c = c[:bs]
                    out = [z, c]  # then this
                    return out
                else:
                    raise NotImplementedError(f'only supports caption conditioning')
        else:
            raise NotImplementedError(f'model.conditioning_key could not be None!')

    def forward(self, batch):
        x, c = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self._forward_to_net(x, c)
        # TODO Do logging here

        return loss, loss_dict


    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample() # should not be done for training.
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            raise NotImplementedError(f'only supports pretrained model for conditioning')
        return c
    # TODO validation handlers.

@register_model
def latent_diffusion_constractor(*args, **kwargs_latent_diffusion):
    return LatentDiffusion(*args, **kwargs_latent_diffusion)