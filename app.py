from flask import Flask, request, jsonify
import warnings
import os

warnings.filterwarnings("ignore")
app = Flask(__name__)

def encode_pdf_to_base64(pdf_file):
    import base64
    with open(pdf_file, "rb") as f:
        pdf_bytes = f.read()
        return base64.b64encode(pdf_bytes).decode('utf-8')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/mba', methods=['POST'])
def market_basket_api():
    try:
        import pandas as pd
        from mba_pipeline import market_basket_pipeline

        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format. Use CSV or Excel.'}), 400
        pdf_file = market_basket_pipeline(df)
        pdf_base64 = encode_pdf_to_base64(pdf_file)
        if os.path.exists(pdf_file):
            os.remove(pdf_file)
        return jsonify({
            "pdf": pdf_base64,
            "message": "Market Basket Analysis report generated successfully"
        }), 200
    except Exception as e:
        print(f"MBA Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/segmentation', methods=['POST'])
def segmentation_api():
    try:
        import pandas as pd
        from segmentation_pipeline import rfm_segmentation_pipeline

        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file format. Use CSV or Excel."}), 400
        pdf_file, segmentation_data = rfm_segmentation_pipeline(df)
        pdf_base64 = encode_pdf_to_base64(pdf_file)
        if os.path.exists(pdf_file):
            os.remove(pdf_file)
        return jsonify({
            "pdf": pdf_base64,
            "clusters": segmentation_data.to_dict(orient="records"),
            "message": "Customer Segmentation report generated successfully"
        }), 200
    except Exception as e:
        print(f"Segmentation Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/demand', methods=['POST'])
def demand_forecast_api():
    try:
        import pandas as pd
        from demand_pipeline import run_demand_pipeline

        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file format. Use CSV or Excel."}), 400
        pdf_file = run_demand_pipeline(df)
        pdf_base64 = encode_pdf_to_base64(pdf_file)
        if os.path.exists(pdf_file):
            os.remove(pdf_file)
        return jsonify({
            "pdf": pdf_base64,
            "message": "Demand classification report generated successfully"
        }), 200
    except Exception as e:
        print(f" Demand Forecast Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5500))
    print(f"Flask app starting on port {port}...")
    app.run(host='0.0.0.0', port=port, threaded=True)
