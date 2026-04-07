from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

paths = {
    "EWMA": "pipeline_checkpoints/anomaly_ewma/",
    "IQR": "pipeline_checkpoints/anomaly_iqr/",
    "Isolation Forest": "pipeline_checkpoints/anomaly_isolation_forest/",
    "LOF": "pipeline_checkpoints/anomaly_lof/",
    "LSTM Autoencoder": "pipeline_checkpoints/anomaly_lstm/",
    "Moving Avg Robust": "pipeline_checkpoints/anomaly_moving_avg_robust/",
    "OCSVM": "pipeline_checkpoints/anomaly_ocsvm/",
    "Percentile": "pipeline_checkpoints/anomaly_percentile/",
    "Prophet": "pipeline_checkpoints/anomaly_prophet/",
    "ARIMA": "pipeline_checkpoints/arima_forecast_results/",
    "Clusters": "pipeline_checkpoints/clusters/",
    "Change Points": "pipeline_checkpoints/change_points/",
    "Composite Structural": "pipeline_checkpoints/anomaly_composite/"
}

for name, path in paths.items():
    print(f"\n=== {name} ===")
    try:
        df = spark.read.parquet(path)
        df.printSchema()
    except Exception as e:
        print(f"Error loading {name}: {e}")