from fastapi import APIRouter, Request

from app.core.state import get_state
from app.schemas.chat import ToolsOut

router = APIRouter()


@router.get("/tools", response_model=ToolsOut)
async def list_tools(request: Request):
    return {"tools": get_state(request).tool_info}
