from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from object_detection.people_counter import PeopleCounter
from args import get_args

app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")

people_counter = PeopleCounter(get_args())

def frame_generator():
    while True:
        frame = people_counter.get_latest_frame()
        if frame is not None:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame") 

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=open("templates/index.html").read())

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=5001)
    finally:
        people_counter.stop()