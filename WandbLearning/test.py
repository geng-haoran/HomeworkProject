from cmath import e
from functools import total_ordering
import wandb
wandb.login()

import random 
total_runs = 5
for run in range(total_runs):
    wandb.init(
        project="test_proj",
        name=f"exp_{run}",
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epoch": 10
        }
    )
    epochs = 10
    offset = random.random()/5
    for epoch in range(2,epochs):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset
        wandb.log({"acc": acc, "loss": loss})
    wandb.finish()