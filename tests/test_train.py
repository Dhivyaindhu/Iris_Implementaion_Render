import os

import train


def test_train_script_saves_model_and_passes_gate(tmp_path):
    path = tmp_path / "models" / "iris.joblib"
    accuracy = train.main(str(path))
    assert accuracy >= train.MIN_ACCURACY
    assert os.path.exists(path)
