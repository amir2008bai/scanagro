import httpx
from starlette.applications import Starlette
from starlette.middleware.body_limit import RequestBodyLimitMiddleware
from starlette.responses import Response
from starlette.routing import Route


async def test_body_limit_without_content_length():
    async def endpoint(request):
        return Response(await request.body())

    app = Starlette(routes=[Route("/", endpoint, methods=["POST"])])
    app.add_middleware(RequestBodyLimitMiddleware, max_body_size=10)

    async def chunks():
        yield b"123456"
        yield b"789012"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post("/", content=chunks())
    assert response.status_code == 413
