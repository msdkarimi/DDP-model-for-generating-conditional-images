from torchvision import transforms

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def get_image_transform(mode):
    if mode == 'train':
        return transforms.Compose([
                                        transforms.Resize((256, 256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5], std=[0.5])  # This scales [0,1] to [-1,1]
                                    ])
    elif mode == 'val':
        return transforms.Compose([
                                        transforms.Resize((256, 256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5], std=[0.5])  # This scales [0,1] to [-1,1]
                                    ])
    else:
        raise ValueError(f"Invalid mode: {mode}")