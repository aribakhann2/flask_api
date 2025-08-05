def rfm_segmentation_pipeline(df, n_clusters=4):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from fpdf import FPDF
    import tempfile
    import os
    from sklearn.metrics import silhouette_score

    df = df.rename(columns={"sale_date": "date", "product_id": "sku"})
    df['date'] = pd.to_datetime(df['date'])

    rfm = df.groupby('client_id').agg({
        'date': lambda x: (df['date'].max() - x.max()).days,
        'sale_id': 'count',
        'total_price': 'sum'
    }).reset_index()
    rfm.columns = ['client_id', 'recency', 'frequency', 'monetary']

    X = rfm[['recency', 'frequency', 'monetary']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm['cluster'] = kmeans.fit_predict(X_scaled)

    cluster_summary = rfm.groupby('cluster')[['recency', 'frequency', 'monetary']].mean()
    labels_map = {}
    silhouette = silhouette_score(X_scaled, rfm['cluster'])
    print(f"Silhouette Score: {silhouette:.3f}")
    for idx, row in cluster_summary.iterrows():
        if row['frequency'] > cluster_summary['frequency'].quantile(0.75):
            labels_map[idx] = "VIP"
        elif row['recency'] > cluster_summary['recency'].quantile(0.75):
            labels_map[idx] = "Lost"
        elif row['frequency'] > cluster_summary['frequency'].median():
            labels_map[idx] = "Loyal"
        else:
            labels_map[idx] = "Occasional"

    rfm['cluster_label'] = rfm['cluster'].map(labels_map)

    freq_plot_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    rec_plot_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name

    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=rfm, x='frequency', y='monetary', hue='cluster_label', palette='viridis')
    plt.title("Frequency vs Monetary Segmentation")
    plt.tight_layout()
    plt.savefig(freq_plot_path)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=rfm, x='recency', y='monetary', hue='cluster_label', palette='viridis')
    plt.title("Recency vs Monetary Segmentation")
    plt.tight_layout()
    plt.savefig(rec_plot_path)
    plt.close()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, "RFM Customer Segmentation Report", ln=1, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(4)
    pdf.cell(200, 10, f"Silhouette Score: {silhouette:.3f}", ln=1)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, "Interpretation: >0.5 = good clustering, 0.2-0.5 = moderate, <0.2 = weak clustering.")
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Cluster Summary (Mean Values):", ln=1)
    pdf.set_font("Arial", size=10)
    for idx, row in cluster_summary.iterrows():
        pdf.cell(200, 8, f"Cluster {idx} ({labels_map[idx]}): "
                         f"Recency={row['recency']:.1f}, Frequency={row['frequency']:.1f}, Monetary={row['monetary']:.1f}", ln=1)
    pdf.ln(5)
    pdf.image(freq_plot_path, x=10, w=180)
    pdf.ln(70)
    pdf.image(rec_plot_path, x=10, w=180)
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Cluster-wise Clients:", ln=1)
    for idx, label in labels_map.items():
        pdf.set_font("Arial", size=11)
        clients = ', '.join(map(str, rfm[rfm['cluster'] == idx]['client_id'].tolist()))
        pdf.multi_cell(0, 8, f"Cluster {idx} ({label}): {clients}")

    pdf_file = "rfm_segmentation_report.pdf"
    pdf.output(pdf_file)

    os.remove(freq_plot_path)
    os.remove(rec_plot_path)

    return pdf_file, rfm
