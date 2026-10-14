'''
Models for API endpoint POST /house/price/predict
'''
from datetime import datetime
import pydantic


class HousePricePredictPostRequest(pydantic.BaseModel):
    '''
    Model for request API endpoint
    '''
    overall_qual: int = pydantic.Field(..., ge=1, le=10, description="Overall quality (1-10)")
    gr_liv_area: int = pydantic.Field(..., gt=0, description="Above ground living area (sq ft)")
    garage_cars: int = pydantic.Field(..., ge=0, le=5, description="Size of garage in car capacity")
    total_bsmt_sf: int = pydantic.Field(..., ge=0, description="Total basement area (sq ft)")
    year_built: int = pydantic.Field(..., ge=1800, le=2025, description="Year built")

    @pydantic.field_validator("total_bsmt_sf")
    @classmethod
    def validate_bsmt_sf(cls, v: int) -> int:
        '''
        total_bsmt_sf validator
        '''
        if v < 0:
            raise ValueError("total_bsmt_sf must be non-negative")
        return v

    @pydantic.field_validator("year_built")
    @classmethod
    def validate_year(cls, v: int) -> int:
        '''
        year_built validator
        '''
        current_year = datetime.now().year
        if v > current_year:
            raise ValueError(f"year_built cannot be in the future (max {current_year})")
        return v


class HousePricePredictPostResponse200(pydantic.BaseModel):
    '''
    Model for response API endpoint
    '''
    result: float
