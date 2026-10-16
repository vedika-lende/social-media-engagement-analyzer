from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import io
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "Live.csv"
dataframe = None
clustered_df = None

def load_default_data():
    global dataframe
    try:
        dataframe = pd.read_csv(DEFAULT_CSV)
    except FileNotFoundError:
        dataframe = None

load_default_data()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/data", methods=["GET", "POST"])
def get_data():
    global dataframe, clustered_df
    if request.method == "POST":
        uploaded = request.files.get("file")
        if not uploaded or not uploaded.filename.lower().endswith(".csv"):
            return jsonify({"error": "Please choose a valid CSV file."}), 400
        try:
            dataframe = pd.read_csv(uploaded)
            clustered_df = None
        except Exception as exc:
            return jsonify({"error": f"Could not read CSV: {exc}"}), 400

    if dataframe is None:
        return jsonify({"error": "No dataset loaded. Upload your CSV, or place Live.csv beside app.py."}), 404

    numeric = dataframe.select_dtypes(include=np.number).columns.tolist()
    preferred = ["num_reactions", "num_comments", "num_shares", "num_likes",
                 "num_loves", "num_wows", "num_hahas", "num_sads", "num_angrys"]
    suggested = [c for c in preferred if c in numeric] or numeric
    preview = dataframe.head(8).replace({np.nan: None}).to_dict(orient="records")
    return jsonify({
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "column_names": dataframe.columns.tolist(),
        "numeric_features": numeric,
        "suggested_features": suggested,
        "preview": preview
    })

@app.route("/api/cluster", methods=["POST"])
def run_clustering():
    global clustered_df
    if dataframe is None:
        return jsonify({"error": "Upload a dataset first."}), 400

    payload = request.get_json(silent=True) or {}
    numeric = dataframe.select_dtypes(include=np.number).columns.tolist()
    features = [c for c in payload.get("features", []) if c in numeric]
    if not features:
        preferred = ["num_reactions", "num_comments", "num_shares", "num_likes",
                     "num_loves", "num_wows", "num_hahas", "num_sads", "num_angrys"]
        features = [c for c in preferred if c in numeric] or numeric

    if len(dataframe) < 2:
        return jsonify({"error": "At least two records are needed."}), 400
    if not features:
        return jsonify({"error": "No numeric columns are available for clustering."}), 400

    k = max(2, min(int(payload.get("k", 3)), 10, len(dataframe)))
    X = dataframe[features].replace([np.inf, -np.inf], np.nan)
    features = [c for c in features if X[c].notna().any()]
    X = X[features]
    if not features:
        return jsonify({"error": "Selected columns contain no valid numeric values."}), 400

    imputer = SimpleImputer(strategy="median")
    X_filled = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    clustered_df = dataframe.copy()
    clustered_df["Cluster"] = labels

    if X_scaled.shape[1] >= 2:
        points = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    else:
        points = np.column_stack([X_scaled[:, 0], np.zeros(len(X_scaled))])

    cluster_counts = pd.Series(labels).value_counts().sort_index()
    summary = []
    for cluster_id in range(k):
        group = clustered_df[clustered_df["Cluster"] == cluster_id]
        means = group[features].mean(numeric_only=True)
        summary.append({
            "cluster": int(cluster_id),
            "count": int(len(group)),
            "means": {col: (None if pd.isna(val) else float(val)) for col, val in means.items()}
        })

    return jsonify({
        "rows": int(len(clustered_df)),
        "features": features,
        "k": k,
        "counts": [{"cluster": int(i), "count": int(cluster_counts.get(i, 0))} for i in range(k)],
        "summary": summary,
        "scatter": [{"x": float(points[i, 0]), "y": float(points[i, 1]), "cluster": int(labels[i])}
                    for i in range(len(labels))]
    })

@app.route("/api/download")
def download_results():
    if clustered_df is None:
        return jsonify({"error": "Run clustering before downloading results."}), 400
    buffer = io.BytesIO()
    clustered_df.to_csv(buffer, index=False, encoding="utf-8-sig")
    buffer.seek(0)
    return send_file(buffer, mimetype="text/csv", as_attachment=True,
                     download_name="facebook_posts_with_clusters.csv")

if __name__ == "__main__":
    app.run(debug=True)
