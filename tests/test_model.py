import pytest

from src.model import (
    CLASS_NAMES,
    load_model,
    load_or_train,
    predict_one,
    save_model,
    train_model,
)


def test_accuracy_above_threshold(trained):
    _, accuracy = trained
    assert accuracy >= 0.90  # quality gate: CI fails if the model gets worse


def test_training_is_reproducible():
    _, acc1 = train_model(random_state=0)
    _, acc2 = train_model(random_state=0)
    assert acc1 == acc2


def test_prediction_is_valid_class(model):
    label, confidence = predict_one(model, [5.1, 3.5, 1.4, 0.2])
    assert label in CLASS_NAMES
    assert 0.0 <= confidence <= 1.0


def test_known_flowers(model):
    assert predict_one(model, [5.1, 3.5, 1.4, 0.2])[0] == "setosa"
    assert predict_one(model, [6.7, 3.0, 5.2, 2.3])[0] == "virginica"


def test_bad_input_raises(model):
    with pytest.raises(ValueError):
        predict_one(model, [1, 2])


def test_save_and_load_gives_same_prediction(model, tmp_path):
    path = tmp_path / "model.joblib"
    save_model(model, path)
    reloaded = load_model(path)
    sample = [6.0, 2.9, 4.5, 1.5]
    assert predict_one(model, sample) == predict_one(reloaded, sample)


def test_load_or_train_trains_when_file_missing(tmp_path):
    model, source = load_or_train(tmp_path / "missing.joblib")
    assert source == "trained_at_startup"
    assert predict_one(model, [5.1, 3.5, 1.4, 0.2])[0] == "setosa"


def test_load_or_train_loads_existing_file(model, tmp_path):
    path = tmp_path / "saved" / "model.joblib"
    save_model(model, path)  # also creates the missing folder
    loaded, source = load_or_train(path)
    assert source == "file"
    assert predict_one(loaded, [6.7, 3.0, 5.2, 2.3])[0] == "virginica"
