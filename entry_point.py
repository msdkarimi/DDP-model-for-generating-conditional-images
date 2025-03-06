import latent_ddpm_trainer.trainer
from utils.build import builder
from configs.train_model_config import get_config_trainer, get_logger_config, get_all_config, get_latent_diffusion_config

def main():

    builder("trainer", *get_all_config(), **get_logger_config())



if __name__ == '__main__':

    main()
    exit(0)