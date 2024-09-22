# Importar bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
import numpy as np
import category_encoders as ce

# Carregar os dados
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Preencher valores nulos
train_df.fillna('unknown', inplace=True)
test_df.fillna('unknown', inplace=True)

# Criar uma nova feature 'car_age' (considerando 'year' como a coluna do ano de fabricação)
train_df['car_age'] = 2024 - train_df['model_year']  # Substituir 2024 pelo ano atual
test_df['car_age'] = 2024 - test_df['model_year']

# Transformação logarítmica no preço para estabilizar a variância
train_df['price_log'] = np.log1p(train_df['price'])  # log1p para lidar com valores pequenos

# Codificar variáveis categóricas usando Target Encoding
columns = ['brand', 'model', 'fuel_type', 'engine', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
encoder = ce.TargetEncoder(cols=columns)

X = train_df.drop(columns=['id', 'price', 'price_log'])  # Remover também 'price_log' pois é o target
y = train_df['price_log']  # Usando a transformação logarítmica do preço

# Dividir em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar Target Encoding
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_val_encoded = encoder.transform(X_val)

# Remover outliers do conjunto de treino com base em 'price_log'
def remove_outliers(df, target_column):
    Q1 = df[target_column].quantile(0.25)
    Q3 = df[target_column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[target_column] >= lower_bound) & (df[target_column] <= upper_bound)]

train_df = remove_outliers(train_df, 'price_log')

# Aplicar RobustScaler no conjunto de treino e validação
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_val_scaled = scaler.transform(X_val_encoded)

# Definir o modelo com regularização adicional
model = XGBRegressor(objective='reg:squarederror', random_state=42)

# Definir os hiperparâmetros para a busca aleatória
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9, 11],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2],
}

# Usar K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Realizar busca aleatória nos hiperparâmetros
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, cv=kf, scoring='neg_mean_absolute_error', random_state=42)
random_search.fit(X_train_scaled, y_train)

# Fazer previsões no conjunto de validação
y_pred = random_search.best_estimator_.predict(X_val_scaled)

# Avaliar o modelo
mae = mean_absolute_error(y_val, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

# Prever os preços para o conjunto de teste
X_test = test_df.drop(columns=['id'])
X_test_encoded = encoder.transform(X_test)
X_test_scaled = scaler.transform(X_test_encoded)
test_preds = random_search.best_estimator_.predict(X_test_scaled)

# Reverter a transformação logarítmica nos preços previstos
test_preds = np.expm1(test_preds)  # Usar expm1 para reverter log1p

# Salvar as previsões em um arquivo csv para submissão
submission = pd.DataFrame({
    'id': test_df['id'],
    'price': test_preds
})
submission.to_csv('submission.csv', index=False)

print("Submissão salva em submission.csv")
