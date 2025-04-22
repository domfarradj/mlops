# src/trainingflow.py

from metaflow import FlowSpec, step, Parameter
import mlflow
import mlflow.sklearn
import pandas as pd

class TrainingFlow(FlowSpec):
    data_path = Parameter("data",
                          default="data/Health_Sleep_Statistics.csv",
                          help="Path to training CSV")
    seed      = Parameter("seed", default=42, help="Random seed")
    cv_folds  = Parameter("cv",   default=5, type=int,
                          help="Number of CV folds")

    @step
    def start(self):
        self.df = pd.read_csv(self.data_path)
        self.next(self.tune)

    @step
    def tune(self):
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV

        # split X/y
        target = "Sleep Quality"
        X = self.df.drop(columns=[target])
        y = self.df[target]

        # identify columns
        num_cols = X.select_dtypes(include=["int64","float64"]).columns
        cat_cols = X.select_dtypes(include=["object","category"]).columns

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), list(num_cols)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(cat_cols)),
        ])

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("clf", RandomForestClassifier(random_state=self.seed)),
        ])

        grid = GridSearchCV(pipeline,
                            {"clf__n_estimators":[50,100],
                             "clf__max_depth":[5,10]},
                            cv=self.cv_folds,
                            n_jobs=-1)
        grid.fit(X, y)

        self.best_model  = grid.best_estimator_
        self.best_params = grid.best_params_
        self.next(self.register)

    @step
    def register(self):
        mlflow.set_tracking_uri("http://localhost:5001")
        with mlflow.start_run():
            mlflow.log_params(self.best_params)
            # <<< log the entire pipeline >>>
            mlflow.sklearn.log_model(
                self.best_model,
                artifact_path="model",
                registered_model_name="MyModel"
            )
        self.next(self.end)

    @step
    def end(self):
        print("✅ Training complete. Pipeline registered as ‘MyModel’ in MLflow.")

if __name__ == "__main__":
    TrainingFlow()