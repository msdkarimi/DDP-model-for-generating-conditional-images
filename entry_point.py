from latent_ddpm_trainer.trainer import Trainer
from utils.build import builder
from configs.train_model_config import get_config_trainer, get_logger_config, get_all_config, get_latent_diffusion_config


def main():
    train_data_loader = None
    val_data_loader = None
    trainer_ojb: Trainer = builder("trainer",
                                   *(*get_all_config(),
                                     train_data_loader,
                                     val_data_loader),
                                   **get_logger_config())
    trainer_ojb.run()


if __name__ == '__main__':

    main()
    exit(0)