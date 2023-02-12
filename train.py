import numpy as np
import torch
import torchmetrics
from sklearn import preprocessing
from torch.utils.data import DataLoader
from pprint import pprint

import dataset
from model import OCRModel
from make_dataset import read_data, split_train_test_data
from predictions_preprocessing import preprocess_prediction, decode_predictions
from model_fit_predict import save_model, save_metrics, train_model, evaluate_model


def run_training():
    image_files, targets, targets_splitted, targets_flattened = read_data(
        "./captcha_images_v2"
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
        test_size=0.1,
        random_state=1,
    )

    train_dataset = dataset.ClassificationDataset(
        image_paths=train_images,
        targets=train_targets,
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)

    test_dataset = dataset.ClassificationDataset(
        image_paths=test_images,
        targets=test_targets,
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)

    model = OCRModel(num_chars=len(label_encoder.classes_))
    # model = OCRModel(num_chars=len(label_encoder.classes_)).to("cuda")

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )

    for epoch in range(130):
        train_loss = train_model(
            model=model, data_loader=train_loader, optimizer=optimizer
        )

        test_preds, test_loss = evaluate_model(model=model, data_loader=test_loader)

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
        )

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
    save_model(model=model, output_model_path="./artifacts/ocr_model.pth")
    save_metrics(metrics, "./artifacts/metrics.json")


if __name__ == "__main__":
    run_training()
