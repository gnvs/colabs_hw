# Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
print("Загрузка данных...")
df = pd.read_csv("parkinsons.data")

# Подготовка данных
X = df.drop(['name', 'status'], axis=1)
y = df['status']

# Разделение на train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Нормализация признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nРазмер обучающей выборки: {X_train.shape[0]}")
print(f"Размер тестовой выборки: {X_test.shape[0]}")
print(f"Количество признаков: {X_train.shape[1]}")

# Оптимизированная модель XGBoost
print("\nОбучение оптимизированной модели XGBoost...")

# Убираем use_label_encoder (устаревший параметр)
model = xgb.XGBClassifier(
    n_estimators=200,           # увеличиваем количество деревьев
    max_depth=6,                # увеличиваем глубину
    learning_rate=0.05,         # уменьшаем скорость обучения
    subsample=0.9,              # используем 90% данных для каждого дерева
    colsample_bytree=0.9,       # используем 90% признаков для каждого дерева
    min_child_weight=3,         # минимальный вес ребенка
    gamma=0.1,                  # минимальное уменьшение потерь для разделения
    reg_alpha=0.1,              # L1 регуляризация
    reg_lambda=1,               # L2 регуляризация
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train_scaled, y_train)

# Предсказания
y_pred = model.predict(X_test_scaled)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*50}")
print(f"Точность модели на тестовой выборке: {accuracy * 100:.2f}%")
print(f"{'='*50}")

# Подробный отчет
print("\nОтчет классификации:")
print(classification_report(y_test, y_pred, target_names=['Здоров (0)', 'Болезнь (1)']))

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
print("\nМатрица ошибок:")
print(cm)

# Визуализация матрицы ошибок
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Здоров', 'Болезнь'],
            yticklabels=['Здоров', 'Болезнь'])
plt.title('Матрица ошибок - XGBoost для обнаружения болезни Паркинсона')
plt.xlabel('Предсказанный статус')
plt.ylabel('Истинный статус')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Важность признаков
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Топ-10 наиболее важных признаков')
plt.xlabel('Важность')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

print("\nТоп-10 наиболее важных признаков:")
print(feature_importance.head(10))

# ROC-AUC Score
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Случайный классификатор')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend()
plt.grid(True)
plt.savefig('roc_curve.png')
plt.show()

# Кросс-валидация
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"\nКросс-валидация (5-fold):")
print(f"Средняя точность: {cv_scores.mean() * 100:.2f}%")
print(f"Стандартное отклонение: {cv_scores.std() * 100:.2f}%")

# Проверка на достижение точности >95%
if accuracy > 0.95:
    print("\n" + "="*50)
    print("Ок! Модель достигла точности выше 95%!")
    print(f"   Ваша точность: {accuracy * 100:.2f}%")
    print("="*50)
else:
    print(f"\nТекущая точность ({accuracy * 100:.2f}%) ниже 95%")
    print("\nПопробуем найти лучшие гиперпараметры через GridSearch...")
    
    # GridSearch для поиска лучших параметров
    print("\nПоиск оптимальных параметров (это может занять несколько минут)...")
    
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 150, 200],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    grid_search = GridSearchCV(
        xgb_model, 
        param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\nЛучшие параметры: {grid_search.best_params_}")
    print(f"Лучшая точность на кросс-валидации: {grid_search.best_score_ * 100:.2f}%")
    
    # Оценка на тестовой выборке
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test_scaled)
    accuracy_best = accuracy_score(y_test, y_pred_best)
    
    print(f"\nТочность лучшей модели на тестовой выборке: {accuracy_best * 100:.2f}%")
    
    if accuracy_best > 0.95:
        print("\n" + "="*50)
        print("Ок! После настройки параметров точность превысила 95%!")
        print(f"   Итоговая точность: {accuracy_best * 100:.2f}%")
        print("="*50)
    else:
        print(f"\nДаже после оптимизации точность ({accuracy_best * 100:.2f}%) не достигла 95%")
        print("\nРекомендации для улучшения:")
        print("1. Попробуйте увеличить размер выборки (больше данных)")
        print("2. Используйте ансамбль моделей (XGBoost + RandomForest)")
        print("3. Примените SMOTE для балансировки классов")
        
        # Дополнительная попытка: RandomForest + XGBoost ансамбль
        print("\nПробуем ансамбль XGBoost + RandomForest...")
        from sklearn.ensemble import VotingClassifier
        from sklearn.ensemble import RandomForestClassifier
        
        rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        ensemble = VotingClassifier(
            estimators=[('xgb', best_model), ('rf', rf_model)],
            voting='soft'
        )
        
        ensemble.fit(X_train_scaled, y_train)
        y_pred_ensemble = ensemble.predict(X_test_scaled)
        accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
        
        print(f"Точность ансамбля: {accuracy_ensemble * 100:.2f}%")
        
        if accuracy_ensemble > 0.95:
            print("\nАнсамбль достиг точности выше 95%!")
        else:
            print(f"\nТочность ансамбля: {accuracy_ensemble * 100:.2f}%")