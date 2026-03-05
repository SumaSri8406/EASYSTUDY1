from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/ai-tutor")
def ai_tutor():
    return render_template("ai_tutor.html")


@app.route("/study-planner")
def study_planner():
    return render_template("study_planner.html")


@app.route("/collaboration")
def collaboration():
    return render_template("collaboration.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
