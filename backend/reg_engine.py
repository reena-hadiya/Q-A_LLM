import os
import numpy as np
import pymupdf
from docx import Document
from PIL import Image
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-3.5-flash"

embedder = SentenceTransformer("all-MiniLM-L6-v2")


doc_chunks = []
page_metadata = []
chunk_embeddings = []

current_file = None
current_file_type = None

def cosine_similarity(v1, v2):

    return np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )

def extract_pdf(file_path):

    chunks = []
    metadata = []

    doc = pymupdf.open(file_path)

    for page_num in range(len(doc)):

        page = doc[page_num]

        text = page.get_text("text").strip()

        if text:

            chunks.append(text)

        else:

            chunks.append(
                f"[Scanned/Visual Page {page_num + 1}]"
            )

        metadata.append({
            "page_number": page_num + 1,
            "page_index": page_num
        })

    doc.close()

    return chunks, metadata

def extract_txt(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        text = file.read()

    return [text], [
        {
            "page_number": 1,
            "page_index": 0
        }
    ]

def extract_docx(file_path):

    document = Document(file_path)

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    return [text], [
        {
            "page_number": 1,
            "page_index": 0
        }
    ]

def load_document(file_path):

    global doc_chunks
    global page_metadata
    global chunk_embeddings
    global current_file
    global current_file_type

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":

        doc_chunks, page_metadata = extract_pdf(file_path)
        current_file_type = "pdf"

    elif extension == ".txt":

        doc_chunks, page_metadata = extract_txt(file_path)
        current_file_type = "txt"

    elif extension == ".docx":

        doc_chunks, page_metadata = extract_docx(file_path)
        current_file_type = "docx"

    elif extension in [".png", ".jpg", ".jpeg"]:

        doc_chunks = ["[Image Document]"]

        page_metadata = [
            {
                "page_number": 1,
                "page_index": 0
            }
        ]

        current_file_type = "image"

    else:

        raise ValueError(
            "Unsupported file type"
        )

    current_file = file_path

    # Create embeddings for text documents
    if current_file_type != "image":

        chunk_embeddings = embedder.encode(
            doc_chunks,
            convert_to_numpy=True
        )

    else:

        chunk_embeddings = []

    return len(doc_chunks)

def search_document(query):

    if not doc_chunks:

        raise ValueError(
            "No document has been uploaded."
        )

    # Images don't use text retrieval
    if current_file_type == "image":

        return {
            "text": "",
            "metadata": page_metadata[0],
            "score": 1.0
        }

    query_embedding = embedder.encode(
        query,
        convert_to_numpy=True
    )

    scores = [
        cosine_similarity(
            query_embedding,
            document_embedding
        )
        for document_embedding in chunk_embeddings
    ]

    best_index = int(np.argmax(scores))

    return {
        "text": doc_chunks[best_index],
        "metadata": page_metadata[best_index],
        "score": float(scores[best_index])
    }

def ask_document(question):

    if not current_file:

        raise ValueError(
            "Please upload a document first."
        )

    if current_file_type == "image":

        image = Image.open(current_file)

        prompt = f"""
You are a document question-answering assistant.

Analyze the uploaded image and answer the user's question.

Question:
{question}

Rules:
- Answer using only information visible in the image.
- If the answer is not available in the image, say:
  "I don't know based on the uploaded document."
- Do not invent information.
"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                prompt,
                image
            ]
        )

        return {
            "answer": response.text,
            "page": 1,
            "score": 1.0
        }

    match = search_document(question)

    page_number = match["metadata"]["page_number"]

    context = match["text"]

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the provided
document context.

Document Context:
-----------------
{context}
-----------------

User Question:
{question}

Rules:

1. Use only the document context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. If the answer is not present in the context,
   say exactly:

"I don't know based on the uploaded document."

5. Give a clear and simple answer.
6. If useful, mention the page number.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return {
        "answer": response.text,
        "page": page_number,
        "score": match["score"]
    }