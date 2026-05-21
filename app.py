from flask import Flask, jsonify, render_template

from services import water_quality

from services import precipitation

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/methodology/phase-1')
def phase_1():
    return render_template('methodology/phase_1.html', stats=water_quality.stats())


@app.route('/methodology/phase-2')
def phase_2():
    return render_template('methodology/phase_2.html', stats=water_quality.stats())


@app.route('/api/dataset/water-quality')
def api_water_quality():
    return jsonify(water_quality.records())

@app.route('/unsupervised/phase-3')
def phase_3():
    return render_template('unsupervised/phase_3.html', stats=precipitation.stats())


@app.route('/api/dataset/precipitation')
def api_precipitation():
    return jsonify(precipitation.records())

if __name__ == '__main__':
    app.run(debug=True)
