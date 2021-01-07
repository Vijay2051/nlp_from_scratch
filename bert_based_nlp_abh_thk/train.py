import pandas
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn import model_selection, metrics
from transformers import AdamW, get_linear_schedule_with_warmup

import config
import dataset
import engine
import models


def run():
    df = pd.read_csv(config.TRAINING_FILE).fillna("none")
    df.columns = ["review", "sentiment"]
    print(df.head())
    df.sentiment = df.sentiment.apply(lambda x: 1 if x == "positive" else 0)
    df_train, df_valid = model_selection.train_test_split(
        df, test_size=0.1, random_state=42, stratify=df.sentiment.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.BERTDataset(
        review=df_train.review.values, target=df_train.sentiment.values
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.BERTDataset(
        review=df_valid.review.values, target=df_valid.sentiment.values
    )

    val_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=4
    )

    model = models.BertBaseUncased()
    model.to(config.DEVICE)

    param_optimizers = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optmization_parameters = [
        {
            "params": [
                p for n, p in param_optimizers if not any(nd in n for nd in no_decay)
            ],
            "wight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizers if any(nd in n for nd in no_decay)
            ],
            "wight_decay": 0.0,
        },
    ]

    no_of_training_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optmization_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=no_of_training_steps
    )

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_dataloader, model, optimizer, config.DEVICE, scheduler)
        outputs, targets = engine.eval_fn(val_dataloader, model, config.DEVICE)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy


if __name__ == "__main__":
    run()
