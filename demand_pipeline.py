def run_demand_pipeline(df, thresholds=None):
    # Lazy imports (moved inside the function)
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import classification_report
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from fpdf import FPDF

    df = df.copy()
    df = df.dropna(subset=['quantity', 'unit_price'])
    df = df.rename(columns={
        "sale_date": "date",
        "product_id": "sku",
    })

    # Ensure date column is datetime if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Encode SKU
    le = LabelEncoder()
    df['sku_encoded'] = le.fit_transform(df['sku'])

    # Feature Engineering
    df['price_elasticity'] = df.groupby('sku')['unit_price'].pct_change() / df.groupby('sku')['quantity'].pct_change()
    df['price_elasticity'].replace([np.inf, -np.inf], 0, inplace=True)
    df['price_elasticity'].fillna(0, inplace=True)
    df['lag_1'] = df.groupby('sku')['quantity'].shift(1).fillna(0)
    df['lag_3'] = df.groupby('sku')['quantity'].shift(3).fillna(0)
    df['rolling_mean_3'] = df.groupby('sku')['quantity'].transform(lambda x: x.rolling(3).mean()).fillna(0)
    df['rolling_std_3'] = df.groupby('sku')['quantity'].transform(lambda x: x.rolling(3).std()).fillna(0)

    # Date Features
    if 'date' in df.columns:
        df['month'] = df['date'].dt.month
        df['weekday'] = df['date'].dt.weekday
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    else:
        df['month'], df['weekday'], df['is_weekend'] = 0, 0, 0

    # SKU Popularity
    df['sku_popularity'] = df['sku'].map(df['sku'].value_counts())

    # Demand Binning
    try:
        df['demand_level'] = pd.qcut(df['quantity'], q=3, labels=[0, 1, 2])
    except:
        df['demand_level'] = pd.cut(df['quantity'], bins=[-1, 1, 10, df['quantity'].max()], labels=[0, 1, 2])

    # Features & Target
    features = ['sku_encoded', 'unit_price', 'price_elasticity', 'lag_1', 'lag_3',
                'rolling_mean_3', 'rolling_std_3', 'month', 'weekday', 'is_weekend', 'sku_popularity']
    X, y = df[features], df['demand_level']

    # Outlier Handling
    for col in ['unit_price', 'price_elasticity', 'lag_1', 'lag_3', 'rolling_mean_3', 'rolling_std_3']:
        X[col] = np.clip(X[col], X[col].quantile(0.01), X[col].quantile(0.99))

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Balance Classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Ensemble Models
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    lgb = LGBMClassifier(random_state=42)
    cat = CatBoostClassifier(verbose=0, random_state=42)
    ensemble = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb), ('lgb', lgb), ('cat', cat)], voting='soft')

    # Train & Predict
    ensemble.fit(X_train, y_train)
    y_proba = ensemble.predict_proba(X_test)

    # Thresholds (Default if None)
    if thresholds is None:
        thresholds = {0: 0.5, 1: 0.55, 2: 0.5}

    y_pred_tuned = []
    for probs in y_proba:
        chosen_class = np.argmax([p / thresholds[i] for i, p in enumerate(probs)])
        y_pred_tuned.append(chosen_class)

    # Classification Report (as text for PDF)
    report_text = classification_report(y_test, y_pred_tuned, digits=3, output_dict=False)

    # Final Predictions for Full Dataset
    df['predicted_demand'] = ensemble.predict(X_scaled)
    df['predicted_label'] = df['predicted_demand'].map({0: 'Low Demand', 1: 'Medium Demand', 2: 'High Demand'})

    # Generate PDF Report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, "Demand Classification Report", ln=1, align='C')
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, report_text)

    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    for level, label in [(2, "High Demand (Top 20%)"), (1, "Medium Demand (Middle 60%)"), (0, "Low Demand (Bottom 20%)")]:
        skus = df[df['predicted_demand'] == level]['sku'].head(20).tolist()
        pdf.cell(200, 8, f"{label}:", ln=1)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 8, ', '.join(map(str, skus)) if skus else "No SKUs")
        pdf.set_font("Arial", size=12)
        pdf.ln(4)

    pdf_file = "demand_classification_report.pdf"
    pdf.output(pdf_file)
    print(f"\n Report saved as {pdf_file}")

    return pdf_file
