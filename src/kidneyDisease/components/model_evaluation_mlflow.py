from src.kidneyDisease.entity.config_entity import EvaluationConfig
from src.kidneyDisease.utils.common import save_json
from pathlib import Path
import tensorflow as tf
import mlflow
import mlflow.keras
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split = 0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation='bilinear'
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)


    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()


    def save_score(self) -> None:
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)


    def log_into_mlflow(self):
        # ---------------- MLflow Logging ----------------

        mlflow.set_tracking_uri(self.config.mlflow_uri)
        
        # Create the experiment if it doesn't exist
        mlflow.set_experiment(self.config.experiment_name)
        artifact_dir = self.config.root_dir
        with mlflow.start_run():

            # Log Parameters
            mlflow.log_params(self.config.all_params)

            # Log Metrics
            mlflow.log_metrics({
                "loss": float(self.score[0]),
                "accuracy": float(self.score[1])
            })

            # Log Keras Model
            mlflow.keras.log_model(
                self.model,
                artifact_path="model"
            )

            # -------------------------------------------------
            # Generate Predictions
            # -------------------------------------------------
            self.valid_generator.reset()

            y_true = self.valid_generator.classes

            y_pred = self.model.predict(
                self.valid_generator,
                verbose=1
            )

            y_pred = y_pred.argmax(axis=1)

            class_names = list(self.valid_generator.class_indices.keys())

            # -------------------------------------------------
            # Confusion Matrix
            # -------------------------------------------------
            cm = confusion_matrix(y_true, y_pred)

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=class_names
            )

            disp.plot(cmap="Blues")
            confusion_matrix_path = os.path.join(
                artifact_dir,
                "confusion_matrix.png"
            )
            plt.savefig(confusion_matrix_path)
            plt.close()
            
            mlflow.log_artifact(confusion_matrix_path)

            # -------------------------------------------------
            # Classification Report
            # -------------------------------------------------
            report = classification_report(
                y_true,
                y_pred,
                target_names=class_names
            )
            classification_report_path = os.path.join(
                artifact_dir,
                "classification_report.txt"
            )
            with open(classification_report_path, "w") as f:
                f.write(report)

            mlflow.log_artifact(classification_report_path)

            # -------------------------------------------------
            # Accuracy Curve (if history exists)
            # -------------------------------------------------
            history_path = os.path.join("artifacts", "training", "history.json")

            if os.path.exists(self.config.history_path):

                with open(history_path, "r") as f:
                    history = json.load(f)

                # Accuracy Curve
                plt.figure(figsize=(8,5))
                plt.plot(history["accuracy"], label="Train")
                plt.plot(history["val_accuracy"], label="Validation")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.legend()

                accuracy_curve_path = os.path.join(
                    artifact_dir,
                    "accuracy_curve.png"
                )
                plt.savefig(accuracy_curve_path)
                plt.close()

                mlflow.log_artifact(accuracy_curve_path)

                # Loss Curve
                plt.figure(figsize=(8,5))
                plt.plot(history["loss"], label="Train")
                plt.plot(history["val_loss"], label="Validation")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()

                loss_curve_path = os.path.join(
                    artifact_dir,
                    "loss_curve.png"
                )
                plt.savefig(loss_curve_path)
                plt.close()

                mlflow.log_artifact(loss_curve_path)

                # Log history.json itself
                mlflow.log_artifact(history_path)

            # -------------------------------------------------
            # Log scores.json if present
            # -------------------------------------------------
            if os.path.exists("scores.json"):
                mlflow.log_artifact("scores.json")

        print("✅ MLflow logging completed successfully!")