import json
from json import JSONDecodeError

import requests
from decouple import config
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["Utility"], prefix='/Utility')


class XMLData(BaseModel):
    xml_content: str


def has_data(props):
    for prop in props.keys():
        if not prop.startswith("z_") and prop != 'count':
            return True
    return False


# this endpoint tries to reduce the data size loaded into a browser.
# sometimes a wps call may return more than 100 MB data which will crush the browser
# most of the returned data is spatial data in JSON format, which
# is not needed for reports. This endpoint only returns non-spatial data from wps.

@router.post("/wps")
async def process_xml(data: XMLData):
    # api_url = config('WPS_URL')
    api_url = 'https://sparcal.sdsc.edu/geoserver/wps'
    response = requests.post(api_url,
                             data=data.xml_content,
                             headers={'Content-Type': 'text/plain',
                                      "Accept": "application/json"})
    try:
        json_data = json.loads(response.text.replace('NaN', '"NaN"'))
        json_data = [feature['properties'] for feature in json_data['features']]
        result = [props for props in json_data if has_data(props)]
        # print(json.dumps(result, indent=3))
        return result
    except JSONDecodeError as error:
        raise HTTPException(status_code=500,
                            detail=f"Encounter an error when parse the return from WPS call: {response.text}")
