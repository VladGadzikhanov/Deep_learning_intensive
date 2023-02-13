import numpy as np
import os
import torch
import hydra
import torchmetrics
from sklearn import preprocessing
from torch.utils.data import DataLoader
from hydra.core.config_store import ConfigStore
from entities import entities
from omegaconf import DictConfig
from pprint import pprint

from data import dataset
from model.model import OCRModel
from data.make_dataset import read_data, split_train_test_data
from model.predictions_preprocessing import (
    preprocess_prediction,
    decode_predictions,
)
from model.model_fit_predict import (
    save_model,
    save_metrics,
    train_model,
    evaluate_model,
    save_predictions,
)

config_store = ConfigStore.instance()
config_store.store(name="config", node=entities.Config)


def get_path(filename):
    return os.path.join(os.getcwd(), filename)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run_training(config: DictConfig):
    image_files, targets, targets_splitted, targets_flattened = read_data(
        config.paths_params.input_data_path
    )

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(targets_flattened)
    targets_encoded = [label_encoder.transform(target) for target in targets_splitted]
    targets_encoded = np.array(targets_encoded) + 1

    (
        train_images,
        test_images,
        train_targets,
        test_targets,
        test_orig_targets,
    ) = split_train_test_data(
        image_files=image_files,
        targets_encoded=targets_encoded,
        targets=targets,
        test_size=config.splitting_params.test_size,
        random_state=config.splitting_params.random_state,
    )

    train_dataset = dataset.ClassificationDataset(
        image_paths=train_images,
        targets=train_targets,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.training_params.batch_size,
        shuffle=True,
    )

    test_dataset = dataset.ClassificationDataset(
        image_paths=test_images,
        targets=test_targets,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.training_params.batch_size,
        shuffle=False,
    )

    model = OCRModel(num_chars=len(label_encoder.classes_)).to(
        config.training_params.device
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training_params.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.training_params.factor,
        patience=config.training_params.patience,
        verbose=True,
    )

    for epoch in range(1, config.training_params.epochs_num + 1):
        train_loss = train_model(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            training_params=config.training_params,
        )

        test_preds, test_loss = evaluate_model(
            model=model, data_loader=test_loader, training_params=config.training_params
        )

        test_decoded_preds = [
            decode_predictions(preds=prediction, encoder=label_encoder)
            for prediction in test_preds
        ]

        test_decoded_preds = [
            preprocess_prediction(prediction)
            for lst in test_decoded_preds
            for prediction in lst
        ]

        char_error_rate = torchmetrics.functional.char_error_rate(
            preds=test_decoded_preds, target=test_orig_targets
        ).item()

        pprint(
            list(
                zip(
                    test_orig_targets,
                    test_decoded_preds,
                )
            )[:6]
        )
        print(f"{epoch=}, {train_loss=}, {test_loss=}, {char_error_rate=}")

        scheduler.step(test_loss)

    metrics = {"test_loss": test_loss, "char_error_rate": char_error_rate}
    save_model(
        model=model, output_model_path=get_path(config.paths_params.output_model_path)
    )
    save_metrics(
        metrics=metrics,
        output_metrics_path=get_path(config.paths_params.output_metrics_path),
    )
    save_predictions(
        test_orig_targets=test_orig_targets,
        test_decoded_preds=test_decoded_preds,
        output_predictions_path=config.paths_params.output_predictions_path,
    )


if __name__ == "__main__":
    run_training()
