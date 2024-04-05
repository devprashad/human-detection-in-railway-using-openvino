from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from ultralytics import YOLO
import imageio
import shutil

app = Flask(__name__, template_folder='templates', static_folder='staticFiles')

model = YOLO('yolov8n.pt', task='detect')


def convert_avi_to_mp4(avi_file, mp4_file):
    reader = imageio.get_reader(avi_file)
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(mp4_file, fps=fps, codec='libx264', quality=8)
    for frame in reader:
        writer.append_data(frame)
    writer.close()

@app.route('/', methods=['GET'])
def render_upload_page():
    return render_template('upload.html')

allowed_image_extensions = {'png', 'jpg', 'jpeg'}
def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_image_extensions

allowed = ['mp4']
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed

@app.route('/upload_pic', methods=['POST'])
def upload_pic():
    if 'image' not in request.files:
        return 'No image selected'
    image = request.files['image']
    if image.filename == "":
        return 'No image file selected'
    if image:
        upload_dir = 'img_data/'
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        global image_path
        image_path = os.path.join(upload_dir, image.filename)
        image.save(image_path)
        print(image_path)
        results = model.predict(upload_dir, save=True)
        for result in results:
            if 'person' in result.names.values():
                return render_template('alert.html')
        output_path = "runs/detect/predict/" + image.filename
        return send_from_directory("runs/detect/predict/", image.filename)
    shutil.rmtree('runs/detect')
    return 'Error uploading image'

@app.route('/upload_sample')
def upload_sample():
    return serve_video2('x2mate.com-How We Can Keep Plastics Out of Our Ocean _ National Geographic.mp4')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return 'No video selected'
    global video
    video = request.files['video']
    if video.filename == "":
        return 'No video file selected'
    if video and allowed_file(video.filename):
        file_name = video.name.split('.')[0]
        upload_dir = 'data/'
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        global video_path
        video_path = os.path.join(upload_dir, video.filename)
        video.save(video_path)
        f_name = video.filename.split('.')[0]
        results = model.predict(video_path, save=True)
        for result in results:
            if 'person' in result.names.values():
                return render_template('alert.html', video_path=video_path)
        mp4_file = 'runs/detect/predict3/' + f_name + '.mp4'
        output_path = "runs/detect/predict/" + f_name + ".avi"
        convert_avi_to_mp4(output_path, mp4_file)
        shutil.rmtree('runs/detect')
        return redirect(url_for('serve_video', filename='result.mp4'))
    shutil.rmtree('runs/detect')
    return 'Error uploading video'

import glob

@app.route('/show_alert', methods=['GET'])
def show_alert():
    # Get the list of all prediction directories
    prediction_dirs = glob.glob('runs/detect/predict*')
    # Sort the directories by creation time to get the latest one
    latest_prediction_dir = max(prediction_dirs, key=os.path.getctime)
    # Get the path of the latest video file in the latest prediction directory
    latest_video_path = os.path.join(latest_prediction_dir, 'video.avi')
    return render_template('alert.html', video_path=url_for('static', filename=latest_video_path))

@app.route('/output')
def output():
    video_name = request.args.get('video_name')
    return render_template('output.html', video_name=video_name)

@app.route('/process', methods=['GET','POST'])
def process():
    return redirect(url_for('success'))

@app.route('/success')
def success():
    # This is the route where the user will be redirected after processing
    return redirect(url_for('serve_video', filename=video.filename))

# Route for serving the uploaded video
@app.route('/<filename>')
def serve_video(filename):
    return send_from_directory('output', filename)

def serve_video2(filename):
    return send_from_directory('sample', filename)

def serve_video3(filename):
    return send_from_directory('runs/detect/predict', filename)

if __name__ == '__main__':
    app.run(debug=True)
