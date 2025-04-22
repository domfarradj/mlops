# src/scoringflow.py

from metaflow import FlowSpec, step, Parameter
import pandas as pd
import mlflow.sklearn

class ScoringFlow(FlowSpec):
    input_path    = Parameter("input",
                              default="data/holdout.csv",
                              help="Path to hold‑out CSV")
    model_version = Parameter("version",
                              default="1",
                              help="Registered model version")

    @step
    def start(self):
        self.df_new = pd.read_csv(self.input_path)
        self.next(self.load_model)

    @step
    def load_model(self):
        mlflow.set_tracking_uri("http://localhost:5001")
        uri = f"models:/MyModel/{self.model_version}"
        self.pipeline = mlflow.sklearn.load_model(uri)
        self.next(self.predict)

    @step
    def predict(self):
        try:
            if hasattr(self.pipeline, "set_output"):
                self.pipeline.set_output(transform="pandas")
            
            # pipeline
            preds = self.pipeline.predict(self.df_new)
            pd.DataFrame({"prediction": preds}) \
            .to_csv("predictions.csv", index=False)
        except ValueError as e:
            # For debugging bc there was some weird error
            print(f"Error: {str(e)}")
         
            
        
        self.next(self.end)

    @step
    def end(self):
        print("✅ Scoring complete. Predictions saved to predictions.csv")

if __name__ == "__main__":
    ScoringFlow()