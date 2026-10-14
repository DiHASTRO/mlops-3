'''
Ping get endpoint models
'''

import pydantic


class PingGetResponse200(pydantic.BaseModel):
    '''
    Ping get response 200 model
    '''
    status: str
