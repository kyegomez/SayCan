import torch
from saycan.model import SayCan

model = SayCan().cuda()

x = torch.randint(0, 256, (1, 1024)).cuda()

model(x) # (1, 1024, 20000)