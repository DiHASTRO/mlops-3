'''
Entry point for whole application
'''
import logging
import pathlib
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

SRC_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, SRC_DIR.as_posix())

from web.api import routes  # pylint: disable=wrong-import-position

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(routes.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
