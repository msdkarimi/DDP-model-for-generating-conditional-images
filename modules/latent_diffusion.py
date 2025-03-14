from modules.gaussian_diffusion import GaussianDiffusion
from modules.clip_text_encoder  import FrozenCLIPEmbedder
from modules.autoencoder import AutoencoderKL
import torch
from utils.diffusion_util import noise_like
from utils.ldm_util import default
from einops import rearrange, repeat
from utils.build import builder, register_model
from modules.distributions import DiagonalGaussianDistribution
from utils.utils import count_params,log_txt_as_img, make_grid
from tqdm import tqdm
from modules.ddim import DDIMSampler



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
                 log_every_t=50
                 ):
        vae_name = kwargs_autoencoder.name
        del kwargs_autoencoder.name
        conditioner_name = kwargs_conditioning_config.name
        del kwargs_conditioning_config.name

        self.conditioning_key = conditioning_key

        self.cond_stage_forward = cond_stage_forward
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs_diffusion_config['timesteps']
        super().__init__(kwargs_unet_config, **kwargs_diffusion_config)
        # side models
        self.first_stage_model = builder(vae_name, **kwargs_autoencoder)
        # self.first_stage_model = builder(vae_name, **kwargs_autoencoder).to(self.device)
        # self.cond_stage_model:FrozenCLIPEmbedder = builder(conditioner_name, **kwargs_conditioning_config).to(self.device)
        self.cond_stage_model:FrozenCLIPEmbedder = builder(conditioner_name, **kwargs_conditioning_config)

        # count_params(self.model, verbose=True)
        # count_params(self.cond_stage_model, verbose=True)
        # count_params(self.first_stage_model, verbose=True)

        self.log_every_t = log_every_t

        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key

        try:
            self.num_downs = len(kwargs_autoencoder.ddconfig.ch_mult) - 1
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
                        clip_denoised: bool,
                        quantize_denoised=False,
                        corrector_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        based on models prediction of noise(aka eps), and given t, tries to predict the x_0, then computes the posterior of mu and var
        """

        model_output = self.model(x, t, c)

        if self.parameterization == 'eps':
            # here we get the x_0
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_output)
        else:
            raise NotImplementedError (f'Currently only supports {self.parameterization} parameterization')

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        """
        Sample x_{t-1} from the model at the given timestep.
        """


        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_recon = self.p_mean_variance(x=x, c=c, t=t,
                                       clip_denoised=clip_denoised,
                                       quantize_denoised=quantize_denoised,
                                       corrector_kwargs=corrector_kwargs)

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x_recon
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise


    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):
        """
        Generate samples from the model, Returns only x_0, Use Case: Standard inference,
        """

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling <t>', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None: # TODO what is mask for
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match


        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            # if self.shorten_cond_schedule:
            #     assert self.model.conditioning_key != 'hybrid'
            #     tc = self.cond_ids[ts].to(cond.device)
            #     cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            # if callback: callback(i)
            # if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

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
        # prefix = 'train' if self.training else 'val'
        prefix = 'train' if self.model.training else 'val'


        if self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError(f'only supports parameterization == {self.parameterization}')

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        # logvar_t = self.logvar[t].to(self.device)
        logvar_t = self.logvar[t]
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
        # if self.model.conditioning_key is not None:
        if self.conditioning_key is not None:
            assert c is not None, 'conditioning key must be provided!'
            if not self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            # if self.shorten_cond_schedule:  # TODO: drop this option
            #     tc = self.cond_ids[t].to(self.device)
            #     c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
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
        # if self.model.conditioning_key is not None:
        if self.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox']:
                    cx = batch[cond_key] # this
                    if bs is not None:
                        cx = cx[:bs]
                    out = [z, cx]  # then this
                    if return_first_stage_outputs:
                        # this will be used in  the logging phase
                        xrec = self.decode_first_stage(z)
                        out.extend([x, xrec])
                    if force_c_encode:
                        # this will be used in  the logging phase
                        c = self.get_learned_conditioning(cx)
                        out.append(c)
                    return out
                else:
                    raise NotImplementedError(f'only supports caption conditioning')
        else:
            raise NotImplementedError(f'model.conditioning_key could not be None!')

    def forward(self, batch):
        x, c = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self._forward_to_net(x, c)

        return loss, loss_dict


    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        z = 1. / self.scale_factor * z
        _x_rec = self.first_stage_model.decode(z) # was this
        return _x_rec

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

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        """
        Generate samples from the model and yield intermediate samples from each timestep of diffusion. return all x_t to x_0, Use Case: Debugging, visualization, and step-by-step analysis
        """
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            # if callback: callback(i)
            # if img_callback: img_callback(img, i)
        return img, intermediates

    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        # for visualization purposes only
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                            force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def log_images(self, batch, N=8, n_row=2, sample=True, ddim_steps=None, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, **kwargs):
        log_every_t = kwargs.get('log_every_t', self.log_every_t)
        use_ddim = ddim_steps is not None
        _log = dict()

        z, xc, x, xrec, c = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           )

        N = batch[self.first_stage_key].shape[0]
        n_row = min(n_row, N)
        _log['inputs'] = x
        _log['reconstructions'] = xrec

        if self.cond_stage_key == "caption":
            xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
            _log["conditionings"] = xc

        if plot_diffusion_rows:
            """
            get intermediates of the noisy images of diffusion model, diffusion forward  
            """
            _diffused_images = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % log_every_t == 0 or t == self.num_timesteps - 1: # TODO check if i need to differentiate between the log_every
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    _diffused_images.append(self.decode_first_stage(z_noisy))


            diffusion_row = torch.stack(_diffused_images)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            _log["diffused_images"] = diffusion_grid

        if sample:
            with self.ema_scope('plotting'):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta) # che ck for the return value, z_denoise_row is intermediates from T to 0,  return_x0=True
                x_samples = self.decode_first_stage(samples)
                _log["samples"] = x_samples
                if plot_denoise_rows:
                    denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                    _log["denoise_row"] = denoise_grid

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation Grid Handler")
            _log["progressive_row"] = prog_row

        return _log


    @torch.no_grad()
    def sample_log(self,cond,batch_size,ddim, ddim_steps,**kwargs):

        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates =ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,cond,verbose=False,**kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates


    def validation_step(self, batch):
        _, loss_dict_no_ema = self(batch)
        with self.ema_scope():
            _, loss_dict_ema = self(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}

        return loss_dict_no_ema, loss_dict_ema
        # self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        # self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)


@register_model
def latent_diffusion_constractor(*args, **kwargs_latent_diffusion):
    return LatentDiffusion(*args, **kwargs_latent_diffusion)