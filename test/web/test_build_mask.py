"""Tests for POST /house/price/predict handler."""
from unittest.mock import patch


def test_predict_handler_success(client):
    """Test successful prediction with valid input."""
    payload = {
        "overall_qual": 7,
        "gr_liv_area": 1800,
        "garage_cars": 2,
        "total_bsmt_sf": 800,
        "year_built": 2005,
    }
    with patch("src.ml.inference.predict") as mock_predict:
        mock_predict.return_value = 250000.0
        response = client.post("/house/price/predict", json=payload)
    assert response.status_code == 200
    assert "result" in response.json()
    assert isinstance(response.json()["result"], float)


def test_predict_handler_missing_field(client):
    """Test validation error when a required field is missing."""
    payload = {
        "overall_qual": 7,
        "gr_liv_area": 1800,
        "garage_cars": 2,
        "total_bsmt_sf": 800,
        # year_built missing
    }
    response = client.post("/house/price/predict", json=payload)
    assert response.status_code == 422


def test_predict_handler_invalid_type(client):
    """Test validation error when field has wrong type."""
    payload = {
        "overall_qual": "seven",  # should be int
        "gr_liv_area": 1800,
        "garage_cars": 2,
        "total_bsmt_sf": 800,
        "year_built": 2005,
    }
    response = client.post("/house/price/predict", json=payload)
    assert response.status_code == 422
