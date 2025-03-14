from utils.build import register_model
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    # def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
    def __init__(self, version="../pretrained/clip_model", device="cuda", max_length=77):
        super().__init__()
        print('clip model loaded from {}'.format(version))
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z, tokens

    def encode(self, text):
        return self(text)



@register_model
def clip_constractor(*args, **kwargs):
    return FrozenCLIPEmbedder(*args, **kwargs)


if __name__ == "__main__":
    from utils.utils import count_params
    txt_encoder = FrozenCLIPEmbedder().cuda()
    count_params(txt_encoder, verbose=True)
    # for i in range(1):
    _txt = 'hi i am just a test'
    z_txt, tokens = txt_encoder.encode(_txt)
    print(z_txt.shape)
    print(tokens)
    print(z_txt)