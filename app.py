from flask import Flask, render_template, request
from analyzer import analyze

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB


@app.route("/")
def home():
    return render_template("index.html", error=None, jd="")


@app.route("/analyze", methods=["POST"])
def run():
    f = request.files.get("resume")
    jd = request.form.get("jd", "").strip()

    def fail(msg):
        return render_template("index.html", error=msg, jd=jd), 400

    if not f or not f.filename:
        return fail("Please choose your resume file.")
    if len(jd) < 30:
        return fail("Please paste the full job description (at least a few lines).")
    try:
        result = analyze(f.filename, f.read(), jd)
    except ValueError as e:
        return fail(str(e))
    except Exception:
        return fail("Sorry, this file could not be read. Try saving it again as PDF or DOCX.")
    if not result["text_ok"]:
        return fail("No text found in this file. It may be a scanned image, and ATS systems cannot read images either. "
                    "Export your resume as a text-based PDF from Word or Google Docs.")
    return render_template("result.html", r=result)


@app.errorhandler(413)
def too_big(_):
    return render_template("index.html", error="File too large (maximum 5 MB).", jd=""), 413


if __name__ == "__main__":
    app.run(debug=True)
