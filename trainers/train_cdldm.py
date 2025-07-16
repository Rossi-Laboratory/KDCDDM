# trainers/train_cdldm.py
import torch
from models.cdldm import CDLDM
from models.vae import Encoder, Decoder
from models.unet import DenoiseUNet
from torch.utils.data import DataLoader

def get_dataloaders():
    return DataLoader(...), DataLoader(...)

def train_cdldm():
    model = CDLDM(
        encoder=Encoder(),
        decoder=Decoder(),
        unet_defect=DenoiseUNet(128),
        unet_circuit=DenoiseUNet(128)
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
