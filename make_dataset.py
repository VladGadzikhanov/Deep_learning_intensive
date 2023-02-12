import pathlib
from sklearn import model_selection


def read_data(input_data_path):
    image_files = pathlib.Path(input_data_path)
    image_files = [str(path) for path in image_files.glob("*.png")]
    targets = [pathlib.Path(x).stem for x in image_files]
    targets_splitted = [list(string) for string in targets]
    targets_flattened = []
    for lst in targets_splitted:
        targets_flattened.extend(lst)

    return image_files, targets, targets_splitted, targets_flattened


def split_train_test_data(
    image_files, targets_encoded, targets, test_size, random_state
):
    (
        train_images,
        test_images,
        train_targets,
        test_targets,
        _,
        test_orig_targets,
    ) = model_selection.train_test_split(
        image_files,
        targets_encoded,
        targets,
        test_size=test_size,
        random_state=random_state,
    )

    return train_images, test_images, train_targets, test_targets, test_orig_targets
