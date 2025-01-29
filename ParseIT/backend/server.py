from flask import Flask, request, jsonify
from docling.document_converter import DocumentConverter
import whisper
import os
from openai import OpenAI

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "./uploads"


os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


whisper_model = whisper.load_model("base")


openai_client = OpenAI()


def summarize_text(text):
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize the following text:\n\n{text}"},
            ],
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error while summarizing: {str(e)}"


@app.route("/process", methods=["POST"])
def process_file():
    """Handle file uploads and process them."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    try:
        if file.filename.lower().endswith((".pdf", ".docx", ".pptx", ".txt")):

            converter = DocumentConverter()
            result = converter.convert(file_path)
            extracted_text = result.document.export_to_text()
        elif file.filename.lower().endswith((".mp3", ".wav", ".m4a")):

            result = whisper_model.transcribe(file_path, fp16=False)
            extracted_text = result["text"]
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        summarized_text = summarize_text(extracted_text)

        return jsonify({"summary": summarized_text})

    except Exception as e:
        return jsonify({"error": f"Failed to process the file: {str(e)}"}), 500

    finally:

        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    app.run(port=4004, debug=True)
