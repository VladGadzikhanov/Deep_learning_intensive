# Final DL course assignment at MADE Big Data Academy
## Usage:
```shell
cd main
# requirements setting
pip install -r requirements.txt
# then you need to change the path to the data in the config.yaml
# run model training
python3 training_pipeline.py hydra.job.chdir=True
```
## Project Organization
```shell
├── README.md
├── __pycache__
├── artifacts
│   ├── 2023-02-13_03-09-36
│   │   ├── metrics.json
│   │   ├── ocr_model.pth
│   │   └── training_pipeline.log
│   └── 2023-02-13_13-59-55
│       ├── metrics.json
│       ├── ocr_model.pth
│       ├── predictions.csv
│       └── training_pipeline.log
├── configs
│   ├── config.yaml
│   └── general
│       └── general.yaml
├── data
│   ├── captcha_images_v2
│   ├── captcha_images_v2.zip
│   └── laba-2.pdf
├── main
│   ├── data
│   │   ├── __pycache__
│   │   │   ├── dataset.cpython-310.pyc
│   │   │   └── make_dataset.cpython-310.pyc
│   │   ├── dataset.py
│   │   └── make_dataset.py
│   ├── entities
│   │   ├── __pycache__
│   │   │   └── entities.cpython-310.pyc
│   │   └── entities.py
│   ├── model
│   │   ├── __pycache__
│   │   │   ├── model.cpython-310.pyc
│   │   │   ├── model_fit_predict.cpython-310.pyc
│   │   │   └── predictions_preprocessing.cpython-310.pyc
│   │   ├── model.py
│   │   ├── model_fit_predict.py
│   │   └── predictions_preprocessing.py
│   └── training_pipeline.py
├── notebooks
│   └── Retrospective.ipynb
└── requirements.txt
```

