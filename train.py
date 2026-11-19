"""Train the model and save it to disk.

Run it at build time (Render build command) so the model file is created with
exactly the same library versions the running API will use.
"""
import os
import sys

from src.model import save_model, train_model

MIN_ACCURACY = 0.90  # quality gate: the build fails if the model is worse than this


def main(path=None):
    path = path or os.environ.get("MODEL_PATH", "models/iris_model.joblib")
    model, accuracy = train_model()
    print(f"Test accuracy: {accuracy:.3f}")
    if accuracy < MIN_ACCURACY:
        sys.exit(f"Accuracy {accuracy:.3f} is below {MIN_ACCURACY}. Refusing to save the model.")
    save_model(model, path)
    print(f"Saved model to {path}")
    return accuracy


if __name__ == "__main__":
    main()
