from fastapi import FastAPI
from src.api.v1.routers.lstm.module import router as lstm_router
from src.api.v1.routers.lstm.startup.lifespan import lifespan as lstm_lifespan
import asyncio
import uvicorn

app = FastAPI()

# mount the sub-application that defines its own lifespan
app.mount('/api/v1/lstm', app=lstm_router)


async def _run():
	# explicitly enter the mounted app's lifespan so it sets its own state
	app.state._lstm_lifespan_cm = lstm_lifespan(lstm_router)
	await app.state._lstm_lifespan_cm.__aenter__()

	config = uvicorn.Config(app=app, host="0.0.0.0", port=8000, reload=False)
	server = uvicorn.Server(config)

	try:
		await server.serve()
	finally:
		cm = getattr(app.state, '_lstm_lifespan_cm', None)
		if cm is not None:
			await cm.__aexit__(None, None, None)


if __name__ == "__main__":
	asyncio.run(_run())