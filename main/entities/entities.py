from dataclasses import dataclass


@dataclass()
class GeneralConfig:
    random_state: int


@dataclass()
class PathsConfig:
    input_data_path: str
    output_model_path: str
    output_metrics_path: str
    output_predictions_path: str


@dataclass()
class SplittingConfig:
    test_size: float
    random_state: int


@dataclass()
class TrainingConfig:
    epochs_num: int
    batch_size: int
    learning_rate: float
    factor: float
    patience: int
    device: str


@dataclass()
class Config:
    general: GeneralConfig
    paths_params: PathsConfig
    splitting_params: SplittingConfig
    training_params: TrainingConfig
