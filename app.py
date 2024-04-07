from args import get_args
from flask import Flask, Response, render_template
from object_detection.people_counter import PeopleCounter

app = Flask(__name__)

@app.after_request
def add_header(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache' 
    response.headers['Expires'] = '0'
    return response


@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = people_counter.get_latest_frame()
            if frame is not None:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    people_counter = PeopleCounter(get_args())
    try:
        app.run(threaded=True, port=5001, debug=False)
    finally:
        people_counter.stop()