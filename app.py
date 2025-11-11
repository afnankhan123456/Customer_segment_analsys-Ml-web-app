from flask import Flask, render_template, request, send_file
import pandas as pd
import os
import joblib
from data_preprocesing import clean_data
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------
# Step 1: Define all absolute paths
# ---------------------------------
BASE_PATH = r"C:\Users\Afnan Khan\Desktop\smartseg-project"

TEMPLATES_PATH = os.path.join(BASE_PATH, "templates")
STATIC_PATH = os.path.join(BASE_PATH, "static")
PIPELINE_PATH = os.path.join(BASE_PATH, "customer_segmentation.pkl")  
OUTPUT_DIR = os.path.join(BASE_PATH, "outputs")

# ---------------------------------
# Step 2: Initialize Flask App
# ---------------------------------
app = Flask(
    __name__,
    template_folder=TEMPLATES_PATH,
    static_folder=STATIC_PATH
)

# ---------------------------------
# Step 3: Home Route
# ---------------------------------
@app.route('/')
def index():
    return render_template('upload.html')

# ---------------------------------
# Step 4: Upload and Process Route
# ---------------------------------
@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Step 1: File Upload
        file = request.files.get('file')
        if not file:
            return "⚠️ Please upload a valid file."

        filename = file.filename
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # Step 2: Clean Data
        df_clean = clean_data(df)

        # Step 3: Load trained pipeline (Scaler + PCA + DBSCAN)
        pipeline = joblib.load(PIPELINE_PATH)

        # Extract components
        scaler = pipeline.named_steps['scaler']
        pca = pipeline.named_steps['pca']
        dbscan = pipeline.named_steps['dbscan']

        # ✅ Step 4: Handle column mismatch automatically
        if hasattr(pipeline, "columns_used"):
            expected_cols = pipeline.columns_used
        else:
            expected_cols = df_clean.columns.tolist()

        # Add missing columns as 0
        for col in expected_cols:
            if col not in df_clean.columns:
                df_clean[col] = 0

        # Drop extra columns not used during training
        df_clean = df_clean[[c for c in df_clean.columns if c in expected_cols]]

        # Step 5: Use pipeline components
        X_scaled = scaler.transform(df_clean)
        X_pca = pca.transform(X_scaled)
        labels = dbscan.fit_predict(X_pca)

        # Step 6: Add results
        df["segment_id"] = labels
        df["outlier_flag"] = (labels == -1)

        # Step 7: Summary
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        num_noise = list(labels).count(-1)
        total = len(labels)
        summary_text = f"""
        ✅ Total Records: {total}<br>
        🧩 Total Clusters (excluding noise): {num_clusters}<br>
        ⚠️ Noise Points: {num_noise}
        """

        # Step 8: Visualization (Scatter Plot)
        plt.figure(figsize=(6, 4))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=20)
        plt.title("Customer Segments (PCA + DBSCAN)")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plot_path = os.path.join(STATIC_PATH, "cluster_plot.png")
        plt.savefig(plot_path)
        plt.close()

        # Step 9: Save CSV Output
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "result.csv")
        df.to_csv(output_path, index=False)

        # Step 10: Render result page
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
# Step 5: Download Route
# ---------------------------------
@app.route('/download/<path:filename>')
def download(filename):
    return send_file(filename, as_attachment=True)

# ---------------------------------
# Step 6: Run Flask App
# ---------------------------------
if __name__ == "__main__":
    app.run(debug=True)
