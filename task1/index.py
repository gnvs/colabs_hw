# Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Загрузка датасета

df = pd.read_csv("fake_news.csv")

print("Первые строки датасета:")
print(df.head())

print("\nИнформация о данных:")
print(df.info())

# Подготовка данных

# Текст новостей
X = df['text']

# Метки (REAL / FAKE)
y = df['label']

# Разделение на train/test

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF векторизация

tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7
)

tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Обучение модели

model = PassiveAggressiveClassifier(max_iter=50)

model.fit(tfidf_train, y_train)

# Предсказания

y_pred = model.predict(tfidf_test)

# Оценка точности

accuracy = accuracy_score(y_test, y_pred)

print("\nТочность модели:")
print(round(accuracy * 100, 2), "%")

# Classification Report

print("\nОтчет классификации:")
print(classification_report(y_test, y_pred))

# Матрица ошибок

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Визуализация Confusion Matrix

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['FAKE','REAL'],
    yticklabels=['FAKE','REAL']
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()

# График распределения новостей

plt.figure(figsize=(6,4))

sns.countplot(x='label', data=df)

plt.title("Distribution of REAL vs FAKE News")
plt.xlabel("News Type")
plt.ylabel("Count")

plt.show()

# Топ слова TF-IDF визуализация

feature_names = tfidf_vectorizer.get_feature_names_out()

tfidf_scores = np.asarray(tfidf_train.mean(axis=0)).ravel()

top_indices = tfidf_scores.argsort()[-20:]

top_words = [feature_names[i] for i in top_indices]
top_scores = tfidf_scores[top_indices]

plt.figure(figsize=(10,5))

plt.barh(top_words, top_scores)

plt.title("Top TF-IDF Words")
plt.xlabel("TF-IDF Score")

plt.show()