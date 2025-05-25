import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from pandas.tseries.offsets import DateOffset
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

# Load the CSV files
df_clientes = pd.read_csv('clientes.csv')
df_transacciones = pd.read_csv('transacciones.csv')

# Merge the dataframes on the common 'id' column
merged_df = pd.merge(df_clientes, df_transacciones, on='id')
cleaned_df = merged_df.drop(["id_estado", "id_municipio", "fecha_alta", "fecha_nacimiento", "actividad_empresarial", "tipo_venta", "giro_comercio", "tipo_persona", "genero"], axis=1)


cleaned_df['fecha'] = pd.to_datetime(cleaned_df['fecha'], format='%m/%d/%Y')
cleaned_df['anio'] = cleaned_df['fecha'].dt.year
cleaned_df['mes'] = cleaned_df['fecha'].dt.month
cleaned_df['dia'] = cleaned_df['fecha'].dt.day

cleaned_df.sort_values(by=['id', 'comercio', 'fecha'], inplace=True)
cleaned_df = cleaned_df.drop('fecha', axis=1)

encoder = LabelEncoder()
cleaned_df['comercio_encoded'] = encoder.fit_transform(cleaned_df['comercio'])

# Step 1: Create a heuristic label for subscriptions
df = cleaned_df.copy()

# Combine into a date
df['fecha'] = pd.to_datetime(dict(year=df['anio'], month=df['mes'], day=df['dia']), errors='coerce')

# Group to calculate features per (id, comercio)
features = df.groupby(['id', 'comercio']).agg({
    'fecha': ['min', 'max', 'nunique', 'count'],
    'monto': ['mean', 'std']
}).reset_index()

features.columns = ['id', 'comercio', 'fecha_min', 'fecha_max', 'fecha_nunique', 'fecha_count', 'monto_mean', 'monto_std']

# Heuristic: If they paid the same comercio ≥3 times on unique dates, it's likely a subscription
features['is_subscription'] = ((features['fecha_count'] >= 3) &
                               (features['monto_std'] < 0.5)).astype(int)

# Merge label back to main df
df = df.merge(features[['id', 'comercio', 'is_subscription']], on=['id', 'comercio'], how='left')

# Drop NA (if any) from label
df = df.dropna(subset=['is_subscription'])

# Step 2: Build a feature set
X = df[['comercio_encoded', 'anio', 'mes', 'dia', 'monto']]
y = df['is_subscription']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Step 3: Train Random Forest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


def predict_next_month_spending(user_df):
    results = {}

    # === Total Monthly Spending Prediction ===
    monthly_spending = user_df.groupby(['anio', 'mes'])['monto'].sum().reset_index()
    monthly_spending = monthly_spending.sort_values(['anio', 'mes'])
    monthly_spending['month_index'] = np.arange(len(monthly_spending))

    X_total = monthly_spending[['month_index']]
    y_total = monthly_spending['monto']
    reg_total = LinearRegression()
    reg_total.fit(X_total, y_total)

    next_month_index = np.array([[monthly_spending['month_index'].max() + 1]])
    predicted_total = reg_total.predict(next_month_index)[0]
    results['total'] = predicted_total

    print(f"📈 Predicted total spending next month: ${predicted_total:.2f}")

    # === Per-Merchant Monthly Spending Prediction ===
    merchants = user_df['comercio'].unique()
    results['per_merchant'] = {}

    for merchant in merchants:
        merchant_df = user_df[user_df['comercio'] == merchant]
        monthly = merchant_df.groupby(['anio', 'mes'])['monto'].sum().reset_index()
        monthly = monthly.sort_values(['anio', 'mes'])
        monthly['month_index'] = np.arange(len(monthly))

        if len(monthly) < 2:
            # Not enough data points for regression
            results['per_merchant'][merchant] = monthly['monto'].iloc[-1]
            continue

        X = monthly[['month_index']]
        y = monthly['monto']
        reg = LinearRegression()
        reg.fit(X, y)

        next_idx = np.array([[monthly['month_index'].max() + 1]])
        pred = reg.predict(next_idx)[0]
        if pred < 0:
            continue
        results['per_merchant'][merchant] = pred

        print(f"🔸 {merchant}: Predicted spending next month: ${pred:.2f}")

    return predicted_total, results


def iconic_expense(user_df):
    unique_counts = user_df['comercio'].value_counts()
    most_unique_commerce = unique_counts.idxmin()
    most_unique_count = unique_counts.min()

    return most_unique_commerce, most_unique_count

def predict_next_month_subscriptions(user_id):
    # Filter transactions for the user
    user_df = df[df['id'] == user_id].copy()
    if user_df.empty:
        print("No transactions found for user:", user_id)
        return pd.DataFrame(), 0.0

    # Predict current subscriptions
    feature_columns = ['comercio_encoded', 'anio', 'mes', 'dia', 'monto']
    user_df['subscription_prediction'] = model.predict(user_df[feature_columns])
    predicted_subs = user_df[user_df['subscription_prediction'] == 1].copy()
    if predicted_subs.empty:
        print("No subscriptions predicted for this user")
        return pd.DataFrame(), 0.0

    # Get most recent date
    most_recent_date = user_df['fecha'].max()
    
    # Cutoff date (3 months back)
    cutoff_date = most_recent_date - DateOffset(months=3)

    # Filter to only subscriptions with activity in the last 3 months
    recent_subs = predicted_subs[predicted_subs['fecha'] >= cutoff_date].copy()
    if recent_subs.empty:
        print("No recent subscriptions found for this user")
        return pd.DataFrame(), 0.0

    # Get next month and year
    last_month = most_recent_date.month
    next_month = last_month + 1 if last_month < 12 else 1
    next_year = most_recent_date.year if next_month != 1 else most_recent_date.year + 1

    # --- Prepare next month subscription candidates with all frequent days ---
    # Get all frequent days (modes) per comercio
    frequent_days = recent_subs.groupby('comercio')['dia'].agg(lambda x: x.mode().tolist()).reset_index()

    # Explode to have one row per frequent day
    frequent_days = frequent_days.explode('dia')

    # Base features (drop duplicates)
    base = recent_subs.drop_duplicates(subset='comercio', keep='first').drop(columns='dia')

    # Merge back to get a row per (comercio, frequent day)
    next_month_candidates = frequent_days.merge(base, on='comercio')
    next_month_candidates['mes'] = next_month
    next_month_candidates['anio'] = next_year

    # Predict subscriptions for next month
    X_next_month = next_month_candidates[feature_columns]
    next_month_candidates['subscription_prediction'] = model.predict(X_next_month)
    next_month_subs = next_month_candidates[next_month_candidates['subscription_prediction'] == 1]
   
    return next_month_subs[['comercio', 'monto', 'anio', 'mes', 'dia']]


def all_predictions(user_id):
    user_df = df[df['id'] == user_id].copy()
    if user_df.empty:
        print("No transactions found for user:", user_id)
        return None

    # Predict next month spending
    total_spending, per_merchant_spending = predict_next_month_spending(user_df)

    # Predict subscriptions
    predicted_subs = predict_next_month_subscriptions(user_id)

    # Iconic expense
    iconic_commerce, iconic_count = iconic_expense(user_df)

    # Convert predicted_subs DataFrame to JSON-safe format
    if isinstance(predicted_subs, pd.DataFrame):
        predicted_subs = predicted_subs.to_dict(orient='records')

    # Convert all int64/float64 to native types
    result = {
        'total_spending': float(total_spending),
        'per_merchant_spending': {
            str(k): float(v) for k, v in per_merchant_spending.get('per_merchant', {}).items()
        },
        'predicted_subs': predicted_subs,
        'iconic_commerce': str(iconic_commerce),
        'iconic_count': int(iconic_count)
    }

    return result
    