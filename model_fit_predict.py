import json

from tqdm import tqdm
import torch


def train_model(model, data_loader, optimizer):
    model.train()
    train_loss = 0

    tk = tqdm(data_loader, total=len(data_loader))
    for data in tk:
        # for key, value in data.items():
        # data[key] = value.to("cuda")

        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(data_loader)


def evaluate_model(model, data_loader):
    model.eval()
    test_loss = 0
    test_preds = []

    with torch.no_grad():
        tk = tqdm(data_loader, total=len(data_loader))
        for data in tk:
            # for key, value in data.items():
            # data[key] = value.to("cuda")

            batch_preds, loss = model(**data)
            test_loss += loss.item()
            test_preds.append(batch_preds)

    return test_preds, test_loss / len(data_loader)


def save_model(model, output_model_path):
    torch.save(model, output_model_path)


def save_metrics(metrics, output_metrics_path):
    with open(output_metrics_path, "w") as metrics_file:
        json.dump(metrics, metrics_file)
