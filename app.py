from flask import Flask, render_template, request, send_file
import pandas as pd
import os
import joblib
from data_preprocesing import clean_data
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------
# FIX: Correct base paths for Render (Linux + Windows)
# ---------------------------------
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

TEMPLATES_PATH = os.path.join(BASE_PATH, "templates")
STATIC_PATH = os.path.join(BASE_PATH, "static")
PIPELINE_PATH = os.path.join(BASE_PATH, "customer_segmentation.pkl")
OUTPUT_DIR = os.path.join(BASE_PATH, "outputs")

# Create static & outputs folder (Render needs this)
os.makedirs(STATIC_PATH, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------
# Initialize Flask App
# ---------------------------------
app = Flask(__name__)

# ---------------------------------
# Home Route
# ---------------------------------
@app.route('/')
def index():
    return render_template('upload.html')

# ---------------------------------
# Upload Route
# ---------------------------------
@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files.get('file')
        if not file:
            return "⚠️ Please upload a valid file."

        filename = file.filename
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # Clean data
        df_clean = clean_data(df)

        # Load pipeline
        pipeline = joblib.load(PIPELINE_PATH)
        scaler = pipeline.named_steps['scaler']
        pca = pipeline.named_steps['pca']
        dbscan = pipeline.named_steps['dbscan']

        # Column handling
        if hasattr(pipeline, "columns_used"):
            expected_cols = pipeline.columns_used
        else:
            expected_cols = df_clean.columns.tolist()

        for col in expected_cols:
            if col not in df_clean.columns:
                df_clean[col] = 0

        df_clean = df_clean[[c for c in df_clean.columns if c in expected_cols]]

        # Predictions
        X_scaled = scaler.transform(df_clean)
        X_pca = pca.transform(X_scaled)
        labels = dbscan.fit_predict(X_pca)

        df["segment_id"] = labels
        df["outlier_flag"] = (labels == -1)

        # Summary
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        num_noise = list(labels).count(-1)
        total = len(labels)

        summary_text = f"""
        ✅ Total Records: {total}<br>
        🧩 Total Clusters: {num_clusters}<br>
        ⚠️ Noise Points: {num_noise}
        """

        # Plot
        plot_path = os.path.join(STATIC_PATH, "cluster_plot.png")
        plt.figure(figsize=(6, 4))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=20)
        plt.title("Customer Segments (PCA + DBSCAN)")
        plt.savefig(plot_path)
        plt.close()

        # Save CSV
        output_path = os.path.join(OUTPUT_DIR, "result.csv")
        df.to_csv(output_path, index=False)

        return render_template(
            "result.html",
            tables=df.head().to_html(classes="data", index=False),
            filename=output_path,
            summary=summary_text,
            plot_image="cluster_plot.png"
        )

    except Exception as e:
        return f"❌ Error: {str(e)}"

# ---------------------------------
# Download Route
# ---------------------------------
@app.route('/download')
def download():
    file_path = os.path.join(OUTPUT_DIR, "result.csv")
    return send_file(file_path, as_attachment=True)

# ---------------------------------
# Run Flask App
# ---------------------------------
if __name__ == "__main__":
    app.run(debug=True)
