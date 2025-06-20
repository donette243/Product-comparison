import pandas as pd
import numpy as np
import random
import torch
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

def set_seed(fixe=True):
    if fixe:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


set_seed(fixe=True)

# Генерация двух каталогов по 60 товаров
типы_товаров = [
    "Смартфон", "Ноутбук", "Планшет", "Наушники", "Телевизор", "Клавиатура",
    "Мышь", "Монитор", "Колонка", "Умные часы", "Веб-камера", "Проектор"
]
бренды = [
    "Samsung", "Apple", "Sony", "Xiaomi", "LG", "Lenovo", "Asus",
    "HP", "Acer", "JBL", "Huawei", "Logitech"
]
модели = [f"модель {i}" for i in range(1, 101)]
описания = [
    "Высокое качество изображения и звука.",
    "Быстрая зарядка и длительная автономность.",
    "Поддержка Wi-Fi и Bluetooth.",
    "Совместимость с современными приложениями.",
    "Эргономичный дизайн и прочный корпус.",
    "Подходит для работы, учёбы и развлечений.",
    "Поддержка Full HD и 4K.",
    "Тонкий корпус и лёгкий вес.",
    "Устойчив к пыли и влаге.",
    "Современный стиль и компактные размеры."
]
атрибуты = [
    "128 ГБ, Android, чёрный", "64 ГБ, iOS, белый", "8 ГБ ОЗУ, SSD 256 ГБ",
    "Bluetooth 5.0, шумоподавление", "HDMI, USB-C, AUX", "4000 мАч батарея",
    "экран 6.5 дюймов", "алюминиевый корпус, 1.3 кг", "IP68 влагозащита", "сканер отпечатков пальцев"
]

# Создать каталог с n уникальными продуктами
def сгенерировать_каталог(n=60):
    данные = []
    использованные_названия = set()
    while len(данные) < n:
        товар = random.choice(типы_товаров)
        бренд = random.choice(бренды)
        модель = random.choice(модели)
        название = f"{товар} {бренд} {модель}"
        if название in использованные_названия:
            continue
        использованные_названия.add(название)
        описание = random.choice(описания)
        атрибут = random.choice(атрибуты)
        данные.append({
            "название": название,
            "описание": описание,
            "атрибуты": атрибут
        })
    return pd.DataFrame(данные)

# Создать столбец объединенного текста
def предобработка(df):
    df = df.fillna("")
    df["полное_описание"] = df["название"] + " " + df["описание"] + " " + df["атрибуты"]
    return df

# Генерация и сохранение
каталог_Ozon = предобработка(сгенерировать_каталог())
каталог_Wildberies = предобработка(сгенерировать_каталог())

каталог_Ozon.to_csv("каталог_Ozon.csv", index=False, encoding='utf-8-sig')
каталог_Wildberies.to_csv("каталог_Wildberies.csv", index=False, encoding='utf-8-sig')
print("Каталоги успешно сгенерированы и сохранены.")

#  Генерация эмбеддингов через локальный BERT 
путь_к_модели = "C:/Users/Donete/Documents/Projets/Drone/projet3/bert_modele/"
модель = SentenceTransformer(путь_к_модели)

вектора_Ozon = модель.encode(каталог_Ozon["полное_описание"].tolist(), convert_to_numpy=True)
вектора_Wildberies = модель.encode(каталог_Wildberies["полное_описание"].tolist(), convert_to_numpy=True)

np.save("вектора_Ozon.npy", вектора_Ozon)
np.save("вектора_Wildberies.npy", вектора_Wildberies)
print("Эмбеддинги успешно созданы и сохранены.")

# Подбор похожих товаров по косинусной схожести
сходства = cosine_similarity(вектора_Ozon, вектора_Wildberies)

пары_совпадений = []
for i in range(len(каталог_Ozon)):
    индекс_лучшего = np.argmax(сходства[i])
    лучший_балл = сходства[i][индекс_лучшего]
    пары_совпадений.append({
        "Товар Ozon": каталог_Ozon.iloc[i]["название"],
        "Товар Wildberies": каталог_Wildberies.iloc[индекс_лучшего]["название"],
        "Сходство": round(лучший_балл, 3)
    })

df_совпадения = pd.DataFrame(пары_совпадений)
df_совпадения.to_csv("совпадения_товаров.csv", index=False, encoding="utf-8-sig")
print("Совпадения успешно найдены и сохранены.")
print(df_совпадения.head())

# Визуализация результатов (гистограмма сходств) 
оценки = df_совпадения["Сходство"]
plt.figure(figsize=(10, 5))
sns.histplot(оценки, bins=20, kde=True)
plt.title("Распределение сходства между товарами Ozon и Wildberies")
plt.xlabel("Косинусное сходство")
plt.ylabel("Количество пар")
plt.grid(True)
plt.savefig("Распределение сходства.png")
plt.show()
