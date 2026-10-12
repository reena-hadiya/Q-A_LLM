import os

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory
)

from werkzeug.utils import secure_filename

from backend.reg_engine import (
    load_document,
    ask_document
)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

BACKEND_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "frontend"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

app = Flask(
    __name__,
    template_folder=FRONTEND_DIR,
    static_folder=FRONTEND_DIR,
    static_url_path="/static"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["MAX_CONTENT_LENGTH"] = (
    20 * 1024 * 1024
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

ALLOWED_EXTENSIONS = {
    "pdf",
    "txt",
    "docx",
    "png",
    "jpg",
    "jpeg"
}

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

@app.route(
    "/frontend/<path:filename>"
)
def frontend(filename):

    return send_from_directory(
        FRONTEND_DIR,
        filename
    )

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_document():

    try:

        # Check whether file exists
        if "document" not in request.files:

            return jsonify({
                "success": False,
                "message":
                    "Please select a document."
            }), 400

        file = request.files["document"]

        # Check filename
        if file.filename == "":

            return jsonify({
                "success": False,
                "message":
                    "No file selected."
            }), 400

        # Check extension
        if not allowed_file(
            file.filename
        ):

            return jsonify({
                "success": False,
                "message":
                    "Unsupported file type."
            }), 400

        # Secure filename
        filename = secure_filename(
            file.filename
        )

        # Create complete upload path
        file_path = os.path.join(
            app.config[
                "UPLOAD_FOLDER"
            ],
            filename
        )

        # Save file
        file.save(
            file_path
        )

        # Load and process document
        chunks = load_document(
            file_path
        )

        return jsonify({
            "success": True,
            "message":
                "Document uploaded successfully.",
            "filename":
                filename,
            "chunks":
                chunks
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message":
                str(e)
        }), 500

@app.route(
    "/ask",
    methods=["POST"]
)
def ask_question():

    try:
        data = request.get_json()

        if not data:

            return jsonify({
                "success": False,
                "message":
                    "Invalid request."
            }), 400

        question = data.get(
            "question",
            ""
        ).strip()

        if not question:

            return jsonify({
                "success": False,
                "message":
                    "Please enter a question."
            }), 400

        # Ask question from document
        result = ask_document(
            question
        )


        return jsonify({
            "success": True,
            "answer":
                result["answer"],
            "page":
                result["page"],
            "score":
                result["score"]
        })


    except Exception as e:

        return jsonify({
            "success": False,
            "message":
                str(e)
        }), 500

if __name__ == "__main__":

    app.run(
        debug=True
    )