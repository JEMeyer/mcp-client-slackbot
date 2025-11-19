from aiohttp import web

from src.core.health import HealthMonitor


def create_health_app(health: HealthMonitor):
    async def handle_health(request):
        healthy = await health.is_healthy()
        report = await health.report()
        status = 200 if healthy else 503
        return web.json_response(report, status=status)

    app = web.Application()
    app.router.add_get("/healthz", handle_health)
    return app
