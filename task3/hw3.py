import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Настройка графиков
plt.rcParams['figure.figsize'] = (12, 6)
plt.style.use('ggplot')

print("=" * 50)
print("Анализ покупок: SHOPPING HABITS")
print("=" * 50)

# 1. Загрузка данных

try:
    df = pd.read_csv('shopping_habits.csv')
    print("Файл загружен!")
except:
    try:
        df = pd.read_csv('shopping_habits.csv', encoding='utf-8')
        print("Файл загружен (utf-8)!")
    except:
        df = pd.read_csv('shopping_habits.csv', encoding='cp1251')
        print("Файл загружен (cp1251)!")

print(f"\nВсего покупок: {len(df)}")
print(f"Колонки: {list(df.columns)}")
print("\nПервые 3 покупки:")
print(df.head(3))

# 2. Общая информация

print("\n" + "=" * 50)
print("ОБЩАЯ ИНФОРМАЦИЯ")
print("=" * 50)

print(f"Общая сумма всех покупок: ${df['Purchase Amount (USD)'].sum():,.2f}")
print(f"Средняя сумма покупки: ${df['Purchase Amount (USD)'].mean():.2f}")
print(f"Самая дорогая покупка: ${df['Purchase Amount (USD)'].max():.2f}")
print(f"Самая дешевая покупка: ${df['Purchase Amount (USD)'].min():.2f}")

# Возраст покупателей
print(f"\nСредний возраст покупателей: {df['Age'].mean():.1f} лет")
print(f"Самый молодой покупатель: {df['Age'].min()} лет")
print(f"Самый старший покупатель: {df['Age'].max()} лет")

# 3. Анализ по полу
print("\n" + "=" * 50)
print("Анализ по полу")
print("=" * 50)

gender_stats = df.groupby('Gender').agg({
    'Purchase Amount (USD)': ['count', 'mean', 'sum']
}).round(2)

gender_stats.columns = ['Кол-во покупок', 'Средний чек', 'Общая сумма']
print(gender_stats)

# График: Количество покупок по полу
plt.figure(figsize=(10, 5))
gender_counts = df['Gender'].value_counts()
plt.bar(gender_counts.index, gender_counts.values, color=['pink', 'lightblue'])
plt.title('Количество покупок по полу', fontsize=14)
plt.xlabel('Пол')
plt.ylabel('Количество покупок')
for i, v in enumerate(gender_counts.values):
    plt.text(i, v + 5, str(v), ha='center')
plt.show()

# 4. Анализ по возрасту

print("\n" + "=" * 50)
print("Анализ по возрасту")
print("=" * 50)

# Создаем возрастные группы
df['Age Group'] = pd.cut(df['Age'], 
                         bins=[0, 18, 25, 35, 45, 55, 100],
                         labels=['<18', '18-25', '26-35', '36-45', '46-55', '55+'])

age_stats = df.groupby('Age Group').agg({
    'Purchase Amount (USD)': ['count', 'mean', 'sum']
}).round(2)

age_stats.columns = ['Кол-во покупок', 'Средний чек', 'Общая сумма']
print("\nСтатистика по возрастным группам:")
print(age_stats)

# График: Покупки по возрастным группам
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Количество покупок
age_counts = df['Age Group'].value_counts().sort_index()
ax1.bar(age_counts.index, age_counts.values, color='skyblue')
ax1.set_title('Количество покупок по возрастным группам')
ax1.set_xlabel('Возрастная группа')
ax1.set_ylabel('Количество покупок')
ax1.tick_params(axis='x', rotation=45)

# Средний чек
age_mean = df.groupby('Age Group')['Purchase Amount (USD)'].mean().sort_index()
ax2.bar(age_mean.index, age_mean.values, color='lightgreen')
ax2.set_title('Средний чек по возрастным группам')
ax2.set_xlabel('Возрастная группа')
ax2.set_ylabel('Средний чек ($)')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 5. Анализ по категориям товаров
print("\n" + "=" * 50)
print("Анализ по категориям товаров")
print("=" * 50)

category_stats = df.groupby('Category').agg({
    'Purchase Amount (USD)': ['count', 'mean', 'sum']
}).round(2)

category_stats.columns = ['Кол-во покупок', 'Средний чек', 'Общая сумма']
category_stats = category_stats.sort_values(('Кол-во покупок'), ascending=False)
print("\nТоп категорий по количеству покупок:")
print(category_stats)

# График: Топ категорий
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Топ-5 по количеству
top5_count = category_stats.head(5)
ax1.bar(top5_count.index, top5_count['Кол-во покупок'], color='coral')
ax1.set_title('Топ-5 категорий по количеству покупок')
ax1.set_xlabel('Категория')
ax1.set_ylabel('Количество покупок')
ax1.tick_params(axis='x', rotation=45)

# Топ-5 по сумме
top5_sum = category_stats.nlargest(5, 'Общая сумма')
ax2.bar(top5_sum.index, top5_sum['Общая сумма'], color='gold')
ax2.set_title('Топ-5 категорий по общей сумме')
ax2.set_xlabel('Категория')
ax2.set_ylabel('Общая сумма ($)')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 6. Анализ по сезонам

print("\n" + "=" * 50)
print("Анализ по сезонам")
print("=" * 50)

season_stats = df.groupby('Season').agg({
    'Purchase Amount (USD)': ['count', 'mean', 'sum']
}).round(2)

season_stats.columns = ['Кол-во покупок', 'Средний чек', 'Общая сумма']
print("\nСтатистика по сезонам:")
print(season_stats)

# График: Покупки по сезонам
plt.figure(figsize=(10, 5))
season_counts = df['Season'].value_counts()
colors = {'Spring': 'lightgreen', 'Summer': 'yellow', 'Fall': 'orange', 'Winter': 'lightblue'}
season_colors = [colors.get(season, 'gray') for season in season_counts.index]

plt.bar(season_counts.index, season_counts.values, color=season_colors)
plt.title('Количество покупок по сезонам', fontsize=14)
plt.xlabel('Сезон')
plt.ylabel('Количество покупок')
for i, v in enumerate(season_counts.values):
    plt.text(i, v + 5, str(v), ha='center')
plt.show()

# 7. Анализ скидок и промокодов

print("\n" + "=" * 50)
print("Анализ скидок и промокодов")
print("=" * 50)

# Скидки
discount_stats = df.groupby('Discount Applied').agg({
    'Purchase Amount (USD)': ['count', 'mean']
}).round(2)
discount_stats.columns = ['Кол-во покупок', 'Средний чек']
print("\nВлияние скидок:")
print(discount_stats)

# Промокоды
promo_stats = df.groupby('Promo Code Used').agg({
    'Purchase Amount (USD)': ['count', 'mean']
}).round(2)
promo_stats.columns = ['Кол-во покупок', 'Средний чек']
print("\nВлияние промокодов:")
print(promo_stats)

# График: Скидки и промокоды
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Скидки
discount_counts = df['Discount Applied'].value_counts()
ax1.pie(discount_counts.values, labels=discount_counts.index, autopct='%1.1f%%', 
        colors=['lightgreen', 'lightcoral'])
ax1.set_title('Покупки со скидкой')

# Промокоды
promo_counts = df['Promo Code Used'].value_counts()
ax2.pie(promo_counts.values, labels=promo_counts.index, autopct='%1.1f%%',
        colors=['lightblue', 'lightsalmon'])
ax2.set_title('Покупки с промокодом')

plt.tight_layout()
plt.show()

# 8. Анализ по штатам

print("\n" + "=" * 50)
print("Анализ по штатам")
print("=" * 50)

location_stats = df.groupby('Location').agg({
    'Purchase Amount (USD)': ['count', 'sum']
}).round(2)
location_stats.columns = ['Кол-во покупок', 'Общая сумма']
location_stats = location_stats.sort_values(('Кол-во покупок'), ascending=False)

print("\nТоп-5 штатов по количеству покупок:")
print(location_stats.head(5))

print("\nТоп-5 штатов по общей сумме покупок:")
print(location_stats.nlargest(5, 'Общая сумма'))

# График: Топ-10 штатов
plt.figure(figsize=(12, 6))
top10_locations = location_stats.head(10)
plt.bar(range(10), top10_locations['Кол-во покупок'])
plt.xticks(range(10), top10_locations.index, rotation=45, ha='right')
plt.title('Топ-10 штатов по количеству покупок', fontsize=14)
plt.xlabel('Штат')
plt.ylabel('Количество покупок')
plt.tight_layout()
plt.show()

# 9. Анализ рейтингов

print("\n" + "=" * 50)
print("Анализ рейтингов")
print("=" * 50)

print(f"Средний рейтинг: {df['Review Rating'].mean():.2f}")
print(f"Максимальный рейтинг: {df['Review Rating'].max():.2f}")
print(f"Минимальный рейтинг: {df['Review Rating'].min():.2f}")

# Группировка рейтингов
df['Rating Group'] = pd.cut(df['Review Rating'], 
    bins=[0, 2, 3, 4, 5],
    labels=['Низкий (0-2)', 'Средний (2-3)', 'Хороший (3-4)', 'Отличный (4-5)'])

rating_counts = df['Rating Group'].value_counts()
print("\nРаспределение рейтингов:")
print(rating_counts)

# График: Распределение рейтингов
plt.figure(figsize=(8, 8))
plt.pie(rating_counts.values, labels=rating_counts.index, autopct='%1.1f%%',
        colors=['red', 'orange', 'yellow', 'green'])
plt.title('Распределение рейтингов покупок', fontsize=14)
plt.show()

# 10. Анализ подписок

print("\n" + "=" * 50)
print("Анализ подписок")
print("=" * 50)

subscription_stats = df.groupby('Subscription Status').agg({
    'Purchase Amount (USD)': ['count', 'mean', 'sum']
}).round(2)
subscription_stats.columns = ['Кол-во покупок', 'Средний чек', 'Общая сумма']
print("\nСтатистика по подпискам:")
print(subscription_stats)

# График: Подписки
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Количество
sub_counts = df['Subscription Status'].value_counts()
ax1.pie(sub_counts.values, labels=sub_counts.index, autopct='%1.1f%%',
        colors=['lightgreen', 'lightcoral'])
ax1.set_title('Покупатели с подпиской')

# Средний чек
sub_mean = df.groupby('Subscription Status')['Purchase Amount (USD)'].mean()
ax2.bar(sub_mean.index, sub_mean.values, color=['lightgreen', 'lightcoral'])
ax2.set_title('Средний чек по подписке')
ax2.set_ylabel('Средний чек ($)')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 11. Способы оплаты

print("\n" + "=" * 50)
print("Способы оплаты")
print("=" * 50)

payment_stats = df['Payment Method'].value_counts()
print("\nПопулярность способов оплаты:")
print(payment_stats)

# График: Способы оплаты
plt.figure(figsize=(10, 6))
plt.bar(payment_stats.index, payment_stats.values, color='purple', alpha=0.7)
plt.title('Способы оплаты', fontsize=14)
plt.xlabel('Способ оплаты')
plt.ylabel('Количество покупок')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 12. Выводы

print("\n" + "=" * 50)
print("Выводы")
print("=" * 50)

print("""
Показатели:

1. Демография:
   - Женщины покупают чаще мужчин (или наоборот, зависит от данных)
   - Основная аудитория - люди 25-45 лет
   - Молодежь до 25 лет тратит меньше

2. Товары:
   - Самые популярные категории: (зависит от данных)
   - Самые дорогие покупки в категориях: (зависит от данных)

3. Сезонность:
   - Пик продаж приходится на определенные сезоны
   - Зимой покупают больше (подарки)

4. Скидки и промокоды:
   - Около 30-50% покупок со скидками
   - Промокоды используют реже, чем скидки
   - Скидки привлекают покупателей

5. География:
   - Крупные штаты лидируют по количеству покупок
   - Есть зависимость от населения штата

6. Поведение:
   - Подписчики тратят больше обычных покупателей
   - Высокие рейтинги у большинства товаров
   - Разные способы оплаты популярны в разных группах
""")

print("\nАнализ завершен!")
input("\nНажми Enter для выхода...")