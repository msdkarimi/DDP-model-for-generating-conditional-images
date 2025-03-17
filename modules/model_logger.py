import matplotlib.pyplot as plt
def print_scheduler():
    _schedulers = ['', '', '', '', '', '', '', '']

    pass

def log_model_gradients(writer, model, step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_scalar(f"Gradients/{name}", param.grad.norm().item(), step)

