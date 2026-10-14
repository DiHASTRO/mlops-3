'''
Code for POST /house/price/predict
'''
import fastapi

from ml import inference
from web.models import build_mask as endpoint_models


async def handler(request: fastapi.Request, body: endpoint_models.HousePricePredictPostRequest):
    """Endpoint to ensure that server is alive"""
    return endpoint_models.HousePricePredictPostResponse200(
        result=inference.predict(
            features={
                "Overall Qual": body.overall_qual,
                "Gr Liv Area": body.gr_liv_area,
                "Garage Cars": body.garage_cars,
                "Total Bsmt SF": body.total_bsmt_sf,
                "Year Built": body.year_built,
            },
        ),
    )
