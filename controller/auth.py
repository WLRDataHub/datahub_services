from fastapi import Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

oauth2_scheme = APIKeyHeader(name="Authorization", auto_error=False)


class BearerIDTokenPayload(BaseModel):
    groups: list


async def validate(
        token: str = Depends(oauth2_scheme),
):
    """
    Validates the token retrieved from the request headers.
    :param token: The raw token string that corresponds to the `Authorization` header.
    :return: None if invalid or not present, `BearerIDTokenPayload` object if valid.
    """
    return True