#!/bin/bash


#создание директорий data, в которую будет загружен датасет, папок train и test, в которых будут разделенные данные
mkdir data test train


#устанавливаем необходимые библиотеки
pip install gdown
pip install scikit-learn
pip install pandas


#последовательно запускаем файлы, согласно заданию
python3 data_creation.py #загружает файл с гугл диска и делит данные на train и test
python3 data_preprocessing.py #предобработка данных
python3 model_preparation.py #обучение модели
python3 model_testing.py #проверка модели на тестовых данных, получение метрики
