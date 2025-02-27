import torch
from utils.build import register_model
import sys
import os
from modules.logger import build_logger


class Trainer(object):
    def __init__(self, name=None, logger_name=None, *, optimizer_config ):
        self.logger = build_logger(logger_name or 'trainer_logger')

        self.model = None
        self.gaussian_diffusion = None


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
def build_trainer(**config):
    return Trainer(**config)
