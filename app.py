from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from services import water_quality
from services import precipitation
from services import logistic_regression
from services import kmeans_irca
from services import lda as lda_service

app = Flask(__name__)

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/methodology/phase-1')
def phase_1():
    return render_template('methodology/phase_1.html', stats=water_quality.stats())


@app.route('/methodology/phase-2')
def phase_2():
    return render_template('methodology/phase_2.html', stats=water_quality.stats())


@app.route('/methodology/phase-3')
def phase_3():
      return render_template(
          'methodology/phase_3.html',
          kmeans_eval=kmeans_irca.detailed_metrics(),
          logistic_stats=logistic_regression.stats(),
          lda_stats=lda_service.stats(),
      )


@app.route('/methodology/phase-4')
def phase_4():
      return render_template(
          'methodology/phase_4.html',
          kmeans_eval=kmeans_irca.detailed_metrics(),
          logistic_stats=logistic_regression.stats(),
          logistic_eval=logistic_regression.detailed_metrics(),
          lda_eval=lda_service.detailed_metrics(),
          lda_stats=lda_service.stats(),
      )


@app.route('/methodology/phase-5', methods=['GET', 'POST'])
def phase_5():
    prediction = None
    form_error = None
    selected = {"department_code": None, "year": None}

    if request.method == 'POST':
        raw_dept = request.form.get('department_code', '').strip()
        raw_year = request.form.get('year', '').strip()
        selected = {"department_code": raw_dept, "year": raw_year}

        valid_codes = {d['code'] for d in logistic_regression.departments()}
        valid_years = set(logistic_regression.years())

        try:
            dept_code = int(raw_dept)
            year      = int(raw_year)
        except (TypeError, ValueError):
            form_error = "Please select both a department and a year."
        else:
            if dept_code not in valid_codes:
                form_error = "Unknown department code."
            elif year not in valid_years:
                form_error = "Year must be one of the trained years."
            else:
                prediction = logistic_regression.predict(dept_code, year)
                selected = {
                    "department_code": dept_code,
                    "year": year,
                }

    return render_template(
        'methodology/phase_5.html',
        departments=logistic_regression.departments(),
        years=logistic_regression.years(),
        prediction=prediction,
        form_error=form_error,
        selected=selected,
        logistic_eval=logistic_regression.detailed_metrics(),
    )


@app.route('/api/dataset/water-quality')
def api_water_quality():
    return jsonify(water_quality.records())


@app.route('/api/dataset/lda-water-quality')
def api_lda_water_quality():
    return jsonify(lda_service.records())


@app.route('/api/dataset/precipitation')
def api_precipitation():
    return jsonify(precipitation.records())


@app.route('/reports/<path:filename>')
def reports_file(filename):
    return send_from_directory(REPORTS_DIR, filename)


if __name__ == '__main__':
    app.run(debug=True)