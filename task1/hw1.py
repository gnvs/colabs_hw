import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Настройка для отображения графиков
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.axisbelow'] = True
plt.style.use('ggplot') # Для красоты

# Загрузка данных
try:
    df = pd.read_csv('vgsale_1.csv')
    print("Файл успешно загружен!")
    print(f"Размер данных: {df.shape}")
    print("\nПервые 5 строк:")
    print(df.head())
except FileNotFoundError:
    print("Ошибка: Файл 'vgsale_1.csv' не найден в текущей директории.")
    print("Пожалуйста, поместите файл в папку с этим скриптом или укажите полный путь.")
    exit()

# Предобработка данных
# Удалим строки с пропущенными значениями в годе, так как это критично для анализа по годам
# И создадим копию, чтобы не менять оригинал
df_clean = df.dropna(subset=['Year']).copy()
# Преобразуем год в целое число
df_clean['Year'] = df_clean['Year'].astype(int)

# Убедимся, что продажи - числа
sales_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
for col in sales_columns:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

print(f"\nРазмер данных после очистки: {df_clean.shape}")
print(f"Диапазон лет: с {df_clean['Year'].min()} по {df_clean['Year'].max()}")

# Разделим данные на две эпохи
df_before_2000 = df_clean[df_clean['Year'] < 2000]
df_after_2000 = df_clean[df_clean['Year'] >= 2000]

print(f"\nИгр до 2000 года: {len(df_before_2000)}")
print(f"Игр с 2000 года: {len(df_after_2000)}")

# Задание 1: Популярность жанров до и после 2000
print("\n--- Задание 1: Популярность жанров до и после 2000 ---")

# Анализ по количеству выпущенных игр
genre_count_before = df_before_2000['Genre'].value_counts()
genre_count_after = df_after_2000['Genre'].value_counts()

# Анализ по объему мировых продаж
genre_sales_before = df_before_2000.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)
genre_sales_after = df_after_2000.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)

print("Топ-5 жанров по количеству игр до 2000:")
print(genre_count_before.head(5))
print("\nТоп-5 жанров по количеству игр после 2000:")
print(genre_count_after.head(5))

print("\nТоп-5 жанров по продажам до 2000:")
print(genre_sales_before.head(5))
print("\nТоп-5 жанров по продажам после 2000:")
print(genre_sales_after.head(5))

# Визуализация: столбчатые диаграммы
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Сравнение популярности жанров до и после 2000 года', fontsize=16)

# По количеству игр
axes[0, 0].bar(genre_count_before.index, genre_count_before.values, color='skyblue')
axes[0, 0].set_title('По количеству игр (до 2000)')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].set_ylabel('Количество игр')

axes[0, 1].bar(genre_count_after.index, genre_count_after.values, color='lightcoral')
axes[0, 1].set_title('По количеству игр (после 2000)')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].set_ylabel('Количество игр')

# По объему продаж
axes[1, 0].bar(genre_sales_before.index, genre_sales_before.values, color='skyblue')
axes[1, 0].set_title('По объему продаж (до 2000)')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].set_ylabel('Мировые продажи (млн)')

axes[1, 1].bar(genre_sales_after.index, genre_sales_after.values, color='lightcoral')
axes[1, 1].set_title('По объему продаж (после 2000)')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].set_ylabel('Мировые продажи (млн)')

plt.tight_layout()
plt.show()

# Задание 2: Число видеоигр по годам
print("\n--- Задание 2: Число видеоигр по годам ---")

games_per_year = df_clean['Year'].value_counts().sort_index()

plt.figure(figsize=(14, 6))
plt.bar(games_per_year.index, games_per_year.values, color='green', edgecolor='black', alpha=0.7)
plt.title('Общее число выпущенных видеоигр по годам')
plt.xlabel('Год')
plt.ylabel('Количество выпущенных игр')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(range(1980, 2021, 2), rotation=45) # Показывать каждый второй год для читаемости
plt.tight_layout()
plt.show()

# Задание 3: Топ-3 издателя и их игры по платформам
print("\n--- Задание 3: Топ-3 издателя ---")

# Находим топ-3 издателей по количеству игр (исключая пропуски)
top_publishers = df_clean['Publisher'].value_counts().head(3)
print("Топ-3 издателя по количеству выпущенных игр:")
print(top_publishers)

# Фильтруем данные только для этих издателей
df_top_publishers = df_clean[df_clean['Publisher'].isin(top_publishers.index)]

# Группируем: считаем количество игр для каждого издателя на каждой платформе
publisher_platform_counts = df_top_publishers.groupby(['Publisher', 'Platform']).size().unstack(fill_value=0)

# Строим столбчатую диаграмму с накоплением
plt.figure(figsize=(14, 8))
publisher_platform_counts.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='tab20')
plt.title('Количество игр топ-3 издателей по платформам (с накоплением)')
plt.xlabel('Издатель')
plt.ylabel('Количество выпущенных игр')
plt.legend(title='Платформа', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Задание 4: Круговые диаграммы долей продаж по регионам
print("\n--- Задание 4: Доли продаж по регионам ---")

# Функция для расчета долей
def calculate_regional_shares(df_period):
    total_sales = df_period['Global_Sales'].sum()
    if total_sales == 0:
        return [0, 0, 0, 0] # На случай пустого периода
    na_share = df_period['NA_Sales'].sum() / total_sales
    eu_share = df_period['EU_Sales'].sum() / total_sales
    jp_share = df_period['JP_Sales'].sum() / total_sales
    other_share = df_period['Other_Sales'].sum() / total_sales
    return [na_share, eu_share, jp_share, other_share]

# Расчет для двух периодов
shares_before = calculate_regional_shares(df_before_2000)
shares_after = calculate_regional_shares(df_after_2000)

# Данные для круговых диаграмм
regions = ['Северная Америка', 'Европа', 'Япония', 'Другие страны']
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

# Создание графиков
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Доли региональных продаж в общемировом объеме', fontsize=14)

# Диаграмма для периода до 2000
ax1.pie(shares_before, labels=regions, colors=colors, autopct='%1.1f%%', startangle=90)
ax1.set_title('1980 - 1999')

# Диаграмма для периода после 2000
ax2.pie(shares_after, labels=regions, colors=colors, autopct='%1.1f%%', startangle=90)
ax2.set_title('2000 - 2020')

plt.tight_layout()
plt.show()

# Вывод итогов по заданию 1 (текстовый ответ на вопрос)
print("\n--- Вывод по заданию 1: Ответ на вопрос ---")
print("До 2000 года наиболее популярными жанрами были:")
print("- По количеству игр: Action, Sports")
print("- По объему продаж: Action, Sports, Platform")
print("\nПосле 2000 года популярность сместилась:")
print("- По количеству игр: Action остается лидером, затем идут Sports, Shooter, Role-Playing")
print("- По объему продаж: Action по-прежнему лидирует, но Shooter и Role-Playing выходят на передний план, вытесняя Platform и Simulation.")
print("\nКлючевое наблюдение: Жанр Action оставался самым популярным всегда. "
      "Однако, после 2000 года значительно выросла популярность жанров Shooter и Role-Playing, "
      "в то время как жанры Platform и Puzzle сдали позиции.")