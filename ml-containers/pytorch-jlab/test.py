import torch
# https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu

torch.cuda.current_device()

torch.cuda.device(0)

torch.cuda.device_count()

torch.cuda.get_device_name(0)

torch.cuda.is_available()
