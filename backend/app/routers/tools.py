from fastapi import APIRouter, Request

from app.schemas.chat import ToolsOut

router = APIRouter()


@router.get("/tools", response_model=ToolsOut)
async def list_tools(request: Request):
    return {"tools": request.app.state.tool_info}
