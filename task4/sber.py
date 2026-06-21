import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Настройка графиков
plt.rcParams['figure.figsize'] = (15, 8)
plt.style.use('ggplot')

print("=" * 60)
print("Анализ акций сбербанка: полосы боллинджера")
print("=" * 60)

# 1. загрузка данных с правильным разделителем

try:
    # Загружаем с разделителем точка с запятой
    df = pd.read_csv('sber.csv', sep=';')
    print("файл загружен")
    print(f"размер данных: {df.shape}")
    print(f"колонки: {list(df.columns)}")
    print("\nпервые 3 строки:")
    print(df.head(3))
except Exception as e:
    print(f"ошибка загрузки: {e}")
    exit()

# 2. Преобразование даты

# Преобразуем дату из формата yyyymmdd
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')

# Устанавливаем дату как индекс
df.set_index('DATE', inplace=True)

# Сортируем по дате
df.sort_index(inplace=True)

print(f"\nпериод данных: с {df.index[0].date()} по {df.index[-1].date()}")
print(f"всего торговых дней: {len(df)}")

# 3. Расчет полос боллинджера

# Параметры индикатора
MA_PERIOD = 20      # период скользящей средней
BOL_WIDTH = 2.0     # ширина полос (стандартное значение)

# Цена закрытия
close_price = df['CLOSE']

# Простая скользящая средняя
sma = close_price.rolling(window=MA_PERIOD).mean()

# Стандартное отклонение
std = close_price.rolling(window=MA_PERIOD).std()

# Полосы боллинджера
bb_upper = sma + (std * BOL_WIDTH)
bb_lower = sma - (std * BOL_WIDTH)

# Создаем dataframe для результатов
bb_df = pd.DataFrame({
    'CLOSE': close_price,
    'SMA': sma,
    'BB_UP': bb_upper,
    'BB_DOWN': bb_lower
})

# Убираем первые ma_period-1 строк с nan
bb_df = bb_df.dropna()

print(f"\nРассчитаны полосы боллинджера (sma={MA_PERIOD}, ширина={BOL_WIDTH})")
print("\nПоследние 5 значений:")
print(bb_df.tail())

# 4. Визуализация

plt.figure(figsize=(16, 9))

# График цены закрытия
plt.plot(bb_df.index, bb_df['CLOSE'], 
         label='цена закрытия', color='black', linewidth=1.5, alpha=0.8)

# Полосы боллинджера
plt.plot(bb_df.index, bb_df['SMA'], 
         label=f'sma ({MA_PERIOD})', color='blue', linewidth=2)
plt.plot(bb_df.index, bb_df['BB_UP'], 
         label=f'верхняя полоса (sma + {BOL_WIDTH}σ)', 
         color='red', linestyle='--', linewidth=1.5)
plt.plot(bb_df.index, bb_df['BB_DOWN'], 
         label=f'нижняя полоса (sma - {BOL_WIDTH}σ)', 
         color='green', linestyle='--', linewidth=1.5)

# Закрашиваем область между полосами
plt.fill_between(bb_df.index, bb_df['BB_UP'], bb_df['BB_DOWN'], 
                 alpha=0.1, color='gray')

# Оформление
plt.title('Полосы боллинджера для акций сбербанка', fontsize=16, fontweight='bold')
plt.xlabel('Дата', fontsize=12)
plt.ylabel('Цена (руб)', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Дополнительный график: только последний год

# Берем последние 252 торговых дня (примерно 1 год)
last_year = bb_df.tail(252)

plt.figure(figsize=(16, 8))

plt.plot(last_year.index, last_year['CLOSE'], 
         label='Цена закрытия', color='black', linewidth=2)
plt.plot(last_year.index, last_year['SMA'], 
         label=f'sma ({MA_PERIOD})', color='blue', linewidth=2)
plt.plot(last_year.index, last_year['BB_UP'], 
         label=f'Верхняя полоса', color='red', linestyle='--', linewidth=1.5)
plt.plot(last_year.index, last_year['BB_DOWN'], 
         label=f'Нижняя полоса', color='green', linestyle='--', linewidth=1.5)

plt.fill_between(last_year.index, last_year['BB_UP'], last_year['BB_DOWN'], 
                 alpha=0.1, color='gray')

plt.title('Полосы боллинджера (последний год)', fontsize=16, fontweight='bold')
plt.xlabel('Дата', fontsize=12)
plt.ylabel('Цена (руб)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. Анализ и торговая стратегия

print("\n" + "=" * 60)
print("Анализ графика и торговая стратегия")
print("=" * 60)

# Текущая ситуация
last = bb_df.iloc[-1]
print(f"\nПоследние данные ({bb_df.index[-1].date()}):")
print(f"Цена закрытия: {last['CLOSE']:.2f}")
print(f"Средняя (sma{MA_PERIOD}): {last['SMA']:.2f}")
print(f"Верхняя граница: {last['BB_UP']:.2f}")
print(f"Нижняя граница: {last['BB_DOWN']:.2f}")

# Определяем текущую позицию цены
if last['CLOSE'] > last['BB_UP']:
    position = "Выше верхней полосы (сильная перекупленность)"
elif last['CLOSE'] < last['BB_DOWN']:
    position = "Ниже нижней полосы (сильная перепроданность)"
elif last['CLOSE'] > last['SMA']:
    position = "Выше средней линии"
else:
    position = "Ниже средней линии"

print(f"\nТекущая ситуация: {position}")

print("\n" + "=" * 60)
print("Предложение по торговле")
print("=" * 60)

print("""
Торговая стратегия на основе полос боллинджера:

1. Сигнал к покупке (long):
   - когда цена касается или уходит ниже нижней полосы - актив перепродан
   - желательно дождаться разворота цены вверх
   - цель: движение к средней линии (sma) или выше
   
   пример: при сильном падении, когда цена пробивает нижнюю границу,
   можно открывать длинную позицию с ожиданием отскока.

2. Сигнал к продаже (short) или фиксации прибыли:
   - когда цена касается или уходит выше верхней полосы - актив перекуплен
   - сигнал к закрытию длинных позиций или открытию коротких
   - цель: движение к средней линии
   
   пример: при сильном росте, когда цена пробивает верхнюю границу,
   стоит зафиксировать прибыль или открыть короткую позицию.

3. Дополнительные сигналы:
   - сужение полос (низкая волатильность) → ждем сильного движения
   - расширение полос (высокая волатильность) → подтверждает тренд

Важные замечания для акций сбербанка:

- на графике видно, что цена часто отскакивает от границ полос
- в периоды сильного тренда цена может долго идти вдоль верхней границы
- лучшие сигналы - когда цена пробивает границу и сразу разворачивается
- рекомендуется использовать в комбинации с анализом объемов

Практический пример:
   при достижении ценой нижней полосы (перепроданность) - покупка
   стоп-лосс: на 2-3% ниже минимума
   тейк-профит: средняя линия или верхняя полоса
""")

# Статистика касаний границ
touches_upper = len(bb_df[bb_df['CLOSE'] >= bb_df['BB_UP']])
touches_lower = len(bb_df[bb_df['CLOSE'] <= bb_df['BB_DOWN']])

print(f"\nСтатистика за весь период:")
print(f"Касаний верхней границы: {touches_upper} раз ({touches_upper/len(bb_df)*100:.1f}%)")
print(f"Касаний нижней границы: {touches_lower} раз ({touches_lower/len(bb_df)*100:.1f}%)")
print(f"Всего торговых дней: {len(bb_df)}")

print("\nАнализ завершен")