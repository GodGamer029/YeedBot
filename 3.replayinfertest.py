
import numpy as np
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


net = SimpleNet()
net.loadFromFile('E:\\replays\\scripts\\network.ptyang')
inp = torch.zeros((1, 21))
out = net(inp)
print(out.item())