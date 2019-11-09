# based on https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu

import torch
print("torch.__version__",torch.__version__)
print("torch.cuda.current_device()",torch.cuda.current_device())
print("torch.cuda.device(0)",torch.cuda.device(0))
print("torch.cuda.device_count(),torch.cuda.device_count())
print("torch.cuda.get_device_name(0)",torch.cuda.get_device_name(0))
print("torch.cuda.is_available()",torch.cuda.is_available())
