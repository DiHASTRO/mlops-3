'''
Testing inference model
'''
from unittest.mock import Mock
import pytest
import numpy as np
from src.ml import inference


def test_predict_returns_float_with_mocked_model(monkeypatch):
    """
    Проверяем, что функция predict() принимает словарь с признаками
    и возвращает число. Реальная модель подменяется на фиктивную,
    которая всегда предсказывает 200000.0.
    """
    dummy_model = Mock()
    dummy_model.predict.return_value = np.array([200000.0])

    monkeypatch.setattr(inference, "_model", None)
    monkeypatch.setattr(inference, "_feature_names", None)
    monkeypatch.setattr("joblib.load", lambda _: dummy_model)

    features = {
        "Overall Qual": 7,
        "Gr Liv Area": 1800,
        "Garage Cars": 2,
        "Total Bsmt SF": 800,
        "Year Built": 2005,
    }
    result = inference.predict(features)

    assert isinstance(result, float)
    assert result == 200000.0

    dummy_model.predict.assert_called_once()
    args, _ = dummy_model.predict.call_args
    X_input = args[0]
    assert X_input.shape == (1, 5)


def test_predict_with_list_instead_of_dict(monkeypatch):
    """
    Проверяем, что predict принимает и список значений в правильном порядке.
    """
    dummy_model = Mock()
    dummy_model.predict.return_value = np.array([150000.0])

    monkeypatch.setattr(inference, "_model", None)
    monkeypatch.setattr(inference, "_feature_names", None)
    monkeypatch.setattr("joblib.load", lambda path: dummy_model)

    features_list = [7, 1800, 2, 800, 2005]
    result = inference.predict(features_list)

    assert isinstance(result, float)
    assert result == 150000.0


def test_predict_raises_error_when_features_missing(monkeypatch):
    """
    Проверяем, что если в словаре не хватает признака, выбрасывается ValueError.
    (Модель даже не должна загружаться, но для простоты мокаем)
    """
    monkeypatch.setattr("joblib.load", Mock())
    features = {"Overall Qual": 7, "Gr Liv Area": 1800}

    with pytest.raises(ValueError, match="Отсутствуют признаки"):
        inference.predict(features)
