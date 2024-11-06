from fastapi import FastAPI
from app.routers import process
import app.config  # To initialize logging

app = FastAPI()

# Include the router
app.include_router(process.router, prefix="/KG", tags=["Knowledge Graph"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
