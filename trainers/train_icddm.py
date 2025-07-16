# trainers/train_icddm.py
import torch
from models.icddm import ICDDM
from models.unet import DenoiseUNet
from torch.utils.data import DataLoader
from loss.diffusion_loss import diffusion_loss

# Placeholder dataset class
def get_dataloaders():
    # Replace with actual dataset loading logic
    return DataLoader(...), DataLoader(...)

def train_icddm():
    model = ICDDM(
        unet_defect=DenoiseUNet(1),
        unet_circuit=DenoiseUNet(1)
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dataloader_d, dataloader_c = get_dataloaders()

    for epoch in range(100):
        for (x_defect, _), (x_circuit, _) in zip(dataloader_d, dataloader_c):
            x_defect, x_circuit = x_defect.cuda(), x_circuit.cuda()
            t = torch.randint(0, model.T, (x_defect.size(0),), device=x_defect.device)
            loss = model.training_step(x_defect, x_circuit, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
