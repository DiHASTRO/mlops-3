'''
All routes for fastapi app
'''

from fastapi import APIRouter

from web.api.handlers import build_mask_post
from web.api.handlers import ping_get

router = APIRouter()


router.add_api_route(
    '/ping',
    ping_get.handler,
    methods=['GET'],
    response_model=ping_get.endpoint_models.PingGetResponse200,
)
router.add_api_route(
    '/house/price/predict',
    build_mask_post.handler,
    methods=['POST'],
    response_model=build_mask_post.endpoint_models.HousePricePredictPostResponse200,
)
