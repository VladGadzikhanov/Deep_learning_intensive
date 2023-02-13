# Финальное задание по курсу DL в MADE Big Data Academy
## Как использовать:
```shell
cd main
# установка зависимостей
pip install -r requirements.txt
# далее необходимо изменить путь до данных в файле конфигурации
# запуск обучения модели
python3 training_pipeline.py hydra.job.chdir=True
```
## Структура проекта
```shell
├── README.md
├── artifacts
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
│   │   ├── dataset.py
│   │   └── make_dataset.py
│   ├── entities
│   │   └── entities.py
│   ├── model
│   │   ├── model.py
│   │   ├── model_fit_predict.py
│   │   └── predictions_preprocessing.py
│   └── training_pipeline.py
├── notebooks
│   └── Retrospective.ipynb
└── requirements.txt
```

