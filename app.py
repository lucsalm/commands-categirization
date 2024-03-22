import io

import librosa
import tensorflow as tf
from flask import Flask, request, render_template, Response

from app.load import LoadData

model = LoadData().load_weights_predict("app/files/checkpoints")
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/command', methods=['POST'])
def get_command():
    audio_file = request.files['audio']
    audio_data = audio_file.read()
    audio_stream = io.BytesIO(audio_data)
    audio, sr = librosa.load(audio_stream, mono=True)
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    stft = LoadData().get_spectrogram(audio_resampled)
    stft_data = tf.expand_dims(stft, axis=0)
    out_label = model.predict(stft_data)
    command_id = tf.argmax(out_label, axis=1).numpy()[0]
    command = LoadData().get_commands().get(command_id)
    return Response(f'Did you said {command}?', 200, content_type='text/plain')


if __name__ == '__main__':
    # Use this for build:
    app.run(host='0.0.0.0', debug=True)
    # Use this for local:
    # app.run(debug=True)