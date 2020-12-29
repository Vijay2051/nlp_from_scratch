from torch._C import dtype
from tqdm import tqdm
import torch
import torch.nn as nn

import config


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)


def train_fn(dataloader, model, optimizer, device, accumulation_steps):
    model.train()

    for bi, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        ids = d["ids"]
        mask = d["mask"]
        token_type_ids = d["token_type_ids"]
        targets = d["targets"]

        ids = ids.to(config.device, dtype=torch.long)
        mask = mask.to(config.device, dtype=torch.long)
        token_type_ids = token_type_ids.to(config.device, dtype=torch.long)
        targets = targets.to(config.device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids, attention_mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()


def eval_fn(model, optimizer, dataloader):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(dataloader), total=len(dataloader)):
            ids = d["ids"]
            mask = d["mask"]
            token_type_ids = d["token_type_ids"]
            targets = d["targets"]

            ids = ids.to(config.device, dtype=torch.long)
            mask = mask.to(config.device, dtype=torch.long)
            token_type_ids = token_type_ids.to(config.device, dtype=torch.long)
            targets = targets.to(config.device, dtype=torch.float)

            outputs = model(ids, attention_mask=mask, token_type_ids=token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

            return fin_outputs, fin_targets
