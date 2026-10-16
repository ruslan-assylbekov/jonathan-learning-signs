from fastapi import FastAPI

app = FastAPI(title="Sign Assistant")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
