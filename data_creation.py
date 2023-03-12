#Создайте python-скрипт, который создает различные наборы данных, описывающие некий процесс (например, изменение дневной температуры). 
#Таких наборов должно быть несколько, в некоторые данные можно включить аномалии или шумы. Можно взять готовый датасет, и выкачать его из интернета. 
#Часть наборов данных должны быть сохранены в папке “train”, другая часть в папке “test”.
import pandas as pd
from sklearn.model_selection import train_test_split
import gdown
import os

#скачиваем csv файл с гугл диска и сохраняем в папке data
gdown.download(id="1waefPsrT7sm5rsRMjDHGCXreY45q9tcY", output="./data/dataset.csv", quiet=False)
#открываем данные в виде датафрейма
df = pd.read_csv('Dataset.csv', delimiter = ',', index_col = 'Unnamed: 0')
#делим данные на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(
    df[['id', 'year', 'code', 'period']], 
    df[['polution_clf']], 
    test_size = 0.20, 
    random_state = 42
)
#сохраняем файлы в папках train и test
X_train.to_csv('train/X_train.csv', index=False)
X_test.to_csv('test/X_val.csv', index=False)
Y_train.to_csv('train/Y_train.csv', index=False)
Y_test.to_csv('test/Y_val.csv', index=False)
