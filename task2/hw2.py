import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Настройка графиков
plt.rcParams['figure.figsize'] = (12, 6)
plt.style.use('ggplot')

print("=" * 50)
print("Анализ данных: IQ по странам")
print("=" * 50)

# 1. Загрузка данных

try:
    df = pd.read_csv('IQ_countries.csv')
    print("Файл загружен!")
    print(f"\nВсего стран: {len(df)}")
    print(f"Колонки: {list(df.columns)}")
    print("\nПервые 3 строки:")
    print(df.head(3))
except Exception as e:
    print(f"Ошибка загрузки файла: {e}")
    exit()

# 2. Основная статистика

print("\n" + "=" * 50)
print("Основная статистика")
print("=" * 50)

# Средний IQ по миру
mean_iq = df['Average IQ'].mean()
print(f"Средний IQ в мире: {mean_iq:.1f}")

# Максимальный и минимальный IQ
max_iq = df.loc[df['Average IQ'].idxmax()]
min_iq = df.loc[df['Average IQ'].idxmin()]
print(f"Страна с самым высоким IQ: {max_iq['Country']} ({max_iq['Average IQ']:.1f})")
print(f"Страна с самым низким IQ: {min_iq['Country']} ({min_iq['Average IQ']:.1f})")

# 3. Анализ по континентам

print("\n" + "=" * 50)
print("Сравнение по континентам")
print("=" * 50)

# Группировка по континентам
continent_stats = df.groupby('Continent')['Average IQ'].agg(['mean', 'min', 'max', 'count'])
continent_stats = continent_stats.round(1)
continent_stats.columns = ['Средний IQ', 'Мин. IQ', 'Макс. IQ', 'Кол-во стран']
print(continent_stats)

# График: Средний IQ по континентам
plt.figure(figsize=(10, 6))
continent_stats['Средний IQ'].sort_values().plot(kind='barh', color='skyblue')
plt.title('Средний IQ по континентам', fontsize=14)
plt.xlabel('Средний IQ')
plt.tight_layout()
plt.show()

# 4. Связь IQ с другими показателями

print("\n" + "=" * 50)
print("Взаимосвязь IQ с другими показателями")
print("=" * 50)

# Выбираем числовые колонки для анализа
numeric_cols = ['Average IQ', 'Literacy Rate', 'Human Development Index', 
                'Mean years of schooling', 'Gross National Income', 'Nobel Prices']

# Удаляем строки с пропущенными значениями для корреляции
df_numeric = df[numeric_cols].dropna()

# Считаем корреляции
correlations = df_numeric.corr()['Average IQ'].sort_values(ascending=False)
print("Корреляция IQ с другими показателями:")
for col, corr in correlations.items():
    if col != 'Average IQ':
        print(f"  • {col}: {corr:.2f}")

# Графики зависимости
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Зависимость IQ от различных факторов', fontsize=16)

# IQ vs Грамотность
if 'Literacy Rate' in df.columns:
    axes[0, 0].scatter(df['Literacy Rate'], df['Average IQ'], alpha=0.6)
    axes[0, 0].set_xlabel('Грамотность (%)')
    axes[0, 0].set_ylabel('Средний IQ')
    axes[0, 0].set_title('IQ vs Грамотность')

# IQ vs Индекс развития
if 'Human Development Index' in df.columns:
    axes[0, 1].scatter(df['Human Development Index'], df['Average IQ'], alpha=0.6, color='green')
    axes[0, 1].set_xlabel('Индекс человеческого развития')
    axes[0, 1].set_ylabel('Средний IQ')
    axes[0, 1].set_title('IQ vs HDI')

# IQ vs Образование
if 'Mean years of schooling' in df.columns:
    axes[1, 0].scatter(df['Mean years of schooling'], df['Average IQ'], alpha=0.6, color='red')
    axes[1, 0].set_xlabel('Средние годы обучения')
    axes[1, 0].set_ylabel('Средний IQ')
    axes[1, 0].set_title('IQ vs Образование')

# IQ vs Доход
if 'Gross National Income' in df.columns:
    axes[1, 1].scatter(np.log1p(df['Gross National Income'].dropna()), 
                       df.loc[df['Gross National Income'].notna(), 'Average IQ'], 
                       alpha=0.6, color='purple')
    axes[1, 1].set_xlabel('log(ВНД на душу населения)')
    axes[1, 1].set_ylabel('Средний IQ')
    axes[1, 1].set_title('IQ vs Доход (лог. шкала)')

plt.tight_layout()
plt.show()

# 5. ТОП-10 стран

print("\n" + "=" * 50)
print("ТОП-10 СТРАН")
print("=" * 50)

# Топ-10 по IQ
top_iq = df.nlargest(10, 'Average IQ')[['Country', 'Average IQ', 'Continent']]
print("\n🏆 Топ-10 стран по IQ:")
print(top_iq.to_string(index=False))

# Топ-10 по грамотности (если есть данные)
if 'Literacy Rate' in df.columns:
    top_lit = df.nlargest(10, 'Literacy Rate')[['Country', 'Literacy Rate', 'Continent']]
    print("\n📚 Топ-10 стран по грамотности:")
    print(top_lit.to_string(index=False))

# Топ-10 по нобелевским премиям
if 'Nobel Prices' in df.columns:
    top_nobel = df.nlargest(10, 'Nobel Prices')[['Country', 'Nobel Prices', 'Continent']]
    print("\n🏅 Топ-10 стран по нобелевским премиям:")
    print(top_nobel.to_string(index=False))

# График: Топ-10 по IQ
plt.figure(figsize=(12, 6))
top10 = df.nlargest(10, 'Average IQ')
plt.bar(range(10), top10['Average IQ'])
plt.xticks(range(10), top10['Country'], rotation=45, ha='right')
plt.title('Топ-10 стран по среднему IQ', fontsize=14)
plt.ylabel('Средний IQ')
plt.tight_layout()
plt.show()

# 6. Выводы

print("\n" + "=" * 50)
print("Основные выводы")
print("=" * 50)

print("""
1. IQ сильно связан с образованием и грамотностью
2. Экономическое развитие важно, но не единственный фактор
3. Европа и Азия лидируют по среднему IQ
4. Нобелевские премии сконцентрированы в развитых странах
""")

print("\nАнализ завершен!")
input("\nНажми Enter для выхода...")