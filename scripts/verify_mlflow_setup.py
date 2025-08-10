import mlflow
from chimera.utils.mlflow_utils import log_reproducibility_passport


if __name__ == "__main__":

    # Inicia un run de MLflow
    with mlflow.start_run(run_name="Test_Reproducibility_Passport"):
        print(f"MLflow run started. Run ID: {mlflow.active_run().info.run_id}")

        # Loguea un parámetro de ejemplo
        mlflow.log_param("test_parameter", "hello_world")

        # Llama a la función para loguear el pasaporte
        log_reproducibility_passport()

        # Loguea una métrica de ejemplo
        mlflow.log_metric("test_metric", 1.0)

    print("MLflow run finished.")