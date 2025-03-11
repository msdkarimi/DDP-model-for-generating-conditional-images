from latent_ddpm_trainer.trainer import Trainer
from utils.build import builder
from data.data_loader import DareDataset
from utils.utils import get_image_transform
from configs.train_model_config import get_config_trainer, get_logger_config, get_all_config, get_latent_diffusion_config
from torch.utils.data import DataLoader


def main():
    train_dataset = DareDataset('data', 'train', transform=get_image_transform('train'))
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
    val_data_loader = None
    trainer: Trainer = builder("trainer",
                                   *(*get_all_config(),
                                     train_data_loader,
                                     val_data_loader),
                                   **get_logger_config())
    trainer.run()

if __name__ == '__main__':
    main()
    exit(0)