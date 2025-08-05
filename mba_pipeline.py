def generate_error_pdf(error_message, file_path="mba_error_report.pdf"):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, "Market Basket Analysis - Error Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, f"An error occurred during processing:\n\n{error_message}")
    pdf.output(file_path)
    return file_path

def generate_pdf_report(rules=None, jaccard_pairs=None, jaccard_triplets=None, graph_path=None, file_path="mba_report.pdf"):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, "Market Basket Analysis Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    if rules is not None and not rules.empty:
        pdf.cell(0, 10, "Top Association Rules (FP-Growth):", ln=True)
        for _, row in rules.head(10).iterrows():
            pdf.multi_cell(0, 8, f"{set(row['antecedents'])} => {set(row['consequents'])} "
                                 f"(support={row['support']:.3f}, conf={row['confidence']:.2f}, lift={row['lift']:.2f})")
    else:
        pdf.cell(0, 10, "No high-confidence association rules found.", ln=True)
    if jaccard_pairs:
        pdf.ln(5)
        pdf.cell(0, 10, "Strong Product Pairs (Jaccard >= 0.8):", ln=True)
        for (p1, p2), sim in jaccard_pairs[:15]:
            pdf.multi_cell(0, 8, f"Products ({p1}, {p2}) -> Similarity={sim:.3f}")
    pdf.output(file_path)
    return file_path

def market_basket_pipeline(df, output_pdf="mba_report.pdf", min_support=0.01, jaccard_threshold=0.8):
    import pandas as pd
    from itertools import combinations
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    import networkx as nx
    import matplotlib.pyplot as plt
    import os
    try:
        transactions = df.groupby('client_id')['product_id'].apply(list).tolist()
        print(f"Prepared {len(transactions)} transactions")
        all_products = sorted(set(df['product_id']))
        basket = pd.DataFrame([{p: (p in txn) for p in all_products} for txn in transactions])
        freq_items = fpgrowth(basket, min_support=min_support, use_colnames=True)
        rules = association_rules(freq_items, metric="lift", min_threshold=1.0)
        rules = rules[rules['confidence'] > 0.8]
        if rules.empty:
            print("No high-confidence FP-Growth rules found.")
        print("Running Jaccard similarity...")
        product_customers = df.groupby('product_id')['client_id'].apply(set)
        jaccard_scores = {}
        for p1, p2 in combinations(product_customers.keys(), 2):
            buyers1, buyers2 = product_customers[p1], product_customers[p2]
            intersection = len(buyers1 & buyers2)
            union = len(buyers1 | buyers2)
            if union > 0:
                score = intersection / union
                if score >= jaccard_threshold:
                    jaccard_scores[(p1, p2)] = score
        jaccard_pairs = sorted(jaccard_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"Found {len(jaccard_pairs)} strong Jaccard pairs")
        pdf_path = generate_pdf_report(
            rules=rules if not rules.empty else None,
            jaccard_pairs=jaccard_pairs,
            file_path=output_pdf
        )
        print(f"Report generated: {pdf_path}")
        return pdf_path
    except Exception as e:
        print(f"Error in MBA pipeline: {e}")
        return generate_error_pdf(str(e), file_path="mba_error_report.pdf")
