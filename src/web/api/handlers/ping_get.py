'''
Code for GET /ping endpoint
'''
import fastapi

from web.models import ping_get as endpoint_models


async def handler(request: fastapi.Request):
    """Endpoint to ensure that server is alive"""
    return endpoint_models.PingGetResponse200(status='ok')
