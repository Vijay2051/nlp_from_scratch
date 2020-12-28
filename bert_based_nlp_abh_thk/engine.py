from tqdm import tqdm


def train_fn(dataloader, model, optimzers, device, accumulation_steps):
    model.train()

    for bi, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        pass


def eval_fn():
    pass
