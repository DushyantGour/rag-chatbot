import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from rag_engine import index_documents, query, reindex

UPLOAD_FOLDER = "documents"
ALLOWED_EXTENSIONS = {"pdf", "docx", "xlsx", "xls"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

print("🚀 Starting RAG Chatbot...")
index_documents()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    try:
        answer, sources = query(user_message)
        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload():
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    uploaded = []
    skipped = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            uploaded.append(filename)
        else:
            skipped.append(file.filename)

    if uploaded:
        reindex()

    return jsonify({
        "uploaded": uploaded,
        "skipped": skipped,
        "message": f"✅ {len(uploaded)} file(s) uploaded and indexed!"
    })

@app.route("/documents", methods=["GET"])
def list_documents():
    files = []
    for f in os.listdir(UPLOAD_FOLDER):
        if f.rsplit(".", 1)[-1].lower() in ALLOWED_EXTENSIONS:
            files.append(f)
    return jsonify({"files": files})

@app.route("/reindex", methods=["POST"])
def reindex_docs():
    try:
        reindex()
        return jsonify({"message": "✅ Documents re-indexed successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, port=5000)