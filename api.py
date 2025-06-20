from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/")
def root():
    return {
        "cwd": os.getcwd(),
        "files": os.listdir()
    }
