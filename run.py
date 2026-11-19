"""Run the API the same way the cloud does: host 0.0.0.0, port from the PORT variable."""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
