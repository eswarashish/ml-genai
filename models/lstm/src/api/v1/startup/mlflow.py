import mlflow

def initialize(enable_system_metrics: bool):
    #mflow.pytorch.auto_log() -> best for pytorch lightning
    mlflow.enable_system_metrics_logging() if enable_system_metrics else None
    
