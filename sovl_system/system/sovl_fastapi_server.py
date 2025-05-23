from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import os
import json
from pydantic import BaseModel
from sovl_api import SOVLAPI

# Initialize FastAPI app
app = FastAPI()

# Load Configuration
CONFIG_FILE = "sovl_system/config.json"
try:
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
except FileNotFoundError:
    raise RuntimeError(f"Configuration file not found: {CONFIG_FILE}")

# Initialize SOVL API
sovl_api = SOVLAPI(config_path=CONFIG_FILE)


# Define Pydantic models for API requests
class SOVLRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_k: int = 50
    do_sample: bool = True


class ConfigUpdateRequest(BaseModel):
    key: str
    value: str


# Serve the HTML frontend
@app.get("/")
def serve_frontend():
    frontend_path = "sovl_system/sovl-webui.html"
    if not os.path.exists(frontend_path):
        raise HTTPException(status_code=404, detail="Frontend file not found")
    return FileResponse(frontend_path)


# API endpoint to get configuration details
@app.get("/api/config")
def get_config():
    return {"config": config}


# API endpoint to update configuration
@app.post("/api/config")
def update_config(new_config: dict = Body(...)):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(new_config, f, indent=4)
        global config
        config = new_config
        return {"message": "Configuration updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {str(e)}")


# API endpoint to interact with the SOVL system
@app.post("/api/sovl")
def handle_sovl_request(request: SOVLRequest):
    try:
        # Use SOVLAPI to send input and get output
        sovl_api.send_input({
            "prompt": request.prompt,
            "max_new_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "top_k": request.top_k,
            "do_sample": request.do_sample
        })
        response = sovl_api.get_output()
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing prompt: {str(e)}")


# Endpoint to tune curiosity parameters
@app.post("/api/sovl/tune-curiosity")
def tune_curiosity(params: dict = Body(...)):
    try:
        # If SOVLAPI exposes this, otherwise fallback to system
        if hasattr(sovl_api.system, "tune_curiosity"):
            sovl_api.system.tune_curiosity(**params)
            return {"message": "Curiosity parameters updated successfully"}
        else:
            raise HTTPException(status_code=501, detail="Curiosity tuning not supported via API.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error tuning curiosity: {str(e)}")


# Endpoint to check system health
@app.get("/api/sovl/health")
def check_health():
    try:
        if hasattr(sovl_api.system, "check_memory_health"):
            sovl_api.system.check_memory_health()
            return {"message": "System health check completed successfully"}
        else:
            raise HTTPException(status_code=501, detail="Health check not supported via API.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during health check: {str(e)}")


# Endpoint to update sleep parameters
@app.post("/api/sovl/set-sleep-params")
def set_sleep_params(params: dict = Body(...)):
    try:
        if hasattr(sovl_api.system, "set_sleep_params"):
            sovl_api.system.set_sleep_params(**params)
            return {"message": "Sleep parameters updated successfully"}
        else:
            raise HTTPException(status_code=501, detail="Sleep parameter update not supported via API.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting sleep parameters: {str(e)}")


# Error handler for 404
@app.exception_handler(404)
def custom_404_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"message": "The resource you are looking for was not found."},
    )


# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
