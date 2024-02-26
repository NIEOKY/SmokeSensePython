from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(df_limpio):
    X = df_limpio.drop('frecuencia_fuma', axis=1)  # Variables predictoras
    y = df_limpio['frecuencia_fuma']  # Variable a predecir

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'n_estimators': 200,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'bootstrap': False
    }
    rf = RandomForestClassifier(**param_grid)
    rf.fit(X_train_scaled, y_train)

    y_pred_rf = rf.predict(X_test_scaled)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)

    print('Accuracy Random Forest:', accuracy_rf)

    return rf, scaler

