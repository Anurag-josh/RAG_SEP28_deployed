import os
import time
import json
import uuid
import atexit
import sys
import logging
import threading
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from config.config import validate_config
from utils.document_loader import extract_text_from_file
from utils.auth import hash_password, verify_password, create_jwt_token, decode_jwt_token


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sourceiq_flask_app")
print("[STARTUP] app.py imported", flush=True)

pipeline = None
_pipeline_lock = threading.Lock()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
CHATS_FILE = os.path.join(BASE_DIR, "cache", "sessions.json")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(CHATS_FILE), exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload
print("[STARTUP] Flask app created", flush=True)

def get_pipeline():
    """Initializes the heavy RAG pipeline once, on the first dependent request."""
    global pipeline
    if pipeline is not None:
        return pipeline

    with _pipeline_lock:
        if pipeline is not None:
            return pipeline
        try:
            validate_config()
            logger.info("[STARTUP] Importing RAG pipeline dependencies on demand.")
            from phase10.web_grounded_rag import WebGroundedRAGPipeline

            logger.info("[STARTUP] Initializing RAG pipeline on demand.")
            pipeline = WebGroundedRAGPipeline()
            if "--clear-cache" in sys.argv:
                pipeline.cache_manager.clear_all_cache()
                logger.info("Persistent RAG cache cleared at startup.")
            logger.info("WebGroundedRAGPipeline successfully initialized.")
        except Exception:
            logger.exception("Failed to initialize RAG Pipeline")
    return pipeline


@atexit.register
def close_pipeline_cache():
    """Close the application-scoped cache connections during process shutdown."""
    if pipeline:
        try:
            pipeline.cache_manager.close()
        except Exception as e:
            logger.warning(f"Failed to close cache manager: {e}")

# Storage for uploaded files in session
uploaded_documents = {}  # file_id -> { filename, text, chunk_count }

def load_chats():
    if os.path.exists(CHATS_FILE):
        try:
            with open(CHATS_FILE, "r", encoding="utf-8") as f:
                chats = json.load(f)
                if isinstance(chats, list):
                    return chats
        except Exception as e:
            logger.error(f"Error loading chats session file: {e}")
    
    # Clean default history: Return a single fresh active new chat
    default_chat = [
        {
            "id": f"chat_{uuid.uuid4().hex[:8]}",
            "title": "New chat",
            "active": True,
            "messages": []
        }
    ]
    save_chats(default_chat)
    return default_chat

def save_chats(chats):
    try:
        with open(CHATS_FILE, "w", encoding="utf-8") as f:
            json.dump(chats, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving chats: {e}")

def get_current_user_info():
    """Extracts user information from Authorization header JWT token if present."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header:
        payload = decode_jwt_token(auth_header)
        if payload:
            return payload
    return None

@app.route("/")
def index():
    return render_template("index.html")

# ── Authentication Endpoints ──────────────────────────────────────────────────

@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    pipeline_instance = get_pipeline()
    if not pipeline_instance:
        return jsonify({"success": False, "error": "System uninitialized"}), 500
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()

    if not username or not email or not password:
        return jsonify({"success": False, "error": "Username, email, and password are required."}), 400

    mongo = pipeline_instance.cache_manager.mongo
    if not mongo.available:
        return jsonify({"success": False, "error": "Database service unavailable. Please try again later."}), 503

    # Check for existing email or username explicitly
    if mongo.get_user_by_email(email):
        return jsonify({"success": False, "error": "An account with this email address already exists. Please sign in."}), 400

    if mongo.get_user_by_username(username):
        return jsonify({"success": False, "error": "This username is already taken. Please choose a different username."}), 400

    hashed = hash_password(password)
    user_id = f"usr_{uuid.uuid4().hex[:12]}"
    created = mongo.create_user(user_id, username, email, hashed)
    if not created:
        return jsonify({"success": False, "error": "Failed to create user account. Please try again."}), 500

    token = create_jwt_token(user_id, username, email)
    return jsonify({
        "success": True,
        "token": token,
        "user": {"user_id": user_id, "username": username, "email": email}
    })

@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    pipeline_instance = get_pipeline()
    if not pipeline_instance:
        return jsonify({"success": False, "error": "System uninitialized"}), 500
    data = request.get_json() or {}
    identifier = data.get("email", "").strip() or data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not identifier or not password:
        return jsonify({"success": False, "error": "Email/username and password are required."}), 400

    mongo = pipeline_instance.cache_manager.mongo
    if not mongo.available:
        return jsonify({"success": False, "error": "Database service unavailable. Please try again later."}), 503

    user = mongo.get_user_by_email(identifier) or mongo.get_user_by_username(identifier)
    if not user:
        return jsonify({"success": False, "error": "No account found with this email/username. Click 'Create Account' to register."}), 404

    if not verify_password(password, user.get("password_hash", "")):
        return jsonify({"success": False, "error": "Incorrect password. Please try again."}), 401

    token = create_jwt_token(user["user_id"], user["username"], user["email"])
    return jsonify({
        "success": True,
        "token": token,
        "user": {"user_id": user["user_id"], "username": user["username"], "email": user["email"]}
    })

@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    user_info = get_current_user_info()
    if not user_info:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    return jsonify({"success": True, "user": user_info})

# ── User History Endpoints ────────────────────────────────────────────────────

@app.route("/api/history", methods=["GET"])
def get_history():
    pipeline_instance = get_pipeline()
    if not pipeline_instance:
        return jsonify({"success": False, "error": "System uninitialized"}), 500
    user_info = get_current_user_info()
    user_id = user_info["user_id"] if user_info else "anonymous"
    history = pipeline_instance.cache_manager.mongo.get_user_history(user_id)
    return jsonify({"success": True, "history": history})

@app.route("/api/history/<history_id>", methods=["GET", "DELETE"])
def history_item_detail(history_id):
    pipeline_instance = get_pipeline()
    if not pipeline_instance:
        return jsonify({"success": False, "error": "System uninitialized"}), 500
    user_info = get_current_user_info()
    if not user_info:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    user_id = user_info["user_id"]

    if request.method == "DELETE":
        deleted = pipeline_instance.cache_manager.mongo.delete_user_history_item(user_id, history_id)
        return jsonify({"success": deleted})
    else:
        item = pipeline_instance.cache_manager.mongo.get_user_history_item(user_id, history_id)
        if item:
            return jsonify({"success": True, "history_item": item})
        return jsonify({"success": False, "error": "History item not found"}), 404

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part in request"}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    file_id = f"doc_{uuid.uuid4().hex[:8]}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
    file.save(save_path)

    res = extract_text_from_file(save_path, filename)
    if res["success"]:
        uploaded_documents[file_id] = {
            "id": file_id,
            "filename": filename,
            "text": res["text"],
            "chunk_count": res["chunk_count"]
        }
        logger.info(f"Successfully processed upload: {filename} ({res['chunk_count']} chunks)")
        return jsonify({
            "success": True,
            "file": {
                "id": file_id,
                "filename": filename,
                "chunk_count": res["chunk_count"]
            }
        })
    else:
        return jsonify({"success": False, "error": res["error"]}), 400

@app.route("/api/chat", methods=["POST"])
def chat():
    pipeline_instance = get_pipeline()
    if not pipeline_instance:
        return jsonify({"success": False, "error": "RAG Pipeline not initialized. Check configuration and Groq API key."}), 500
        
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    chat_id = data.get("chat_id")
    if not chat_id or str(chat_id).strip() in ("", "None", "null"):
        chat_id = f"chat_{uuid.uuid4().hex[:8]}"
    file_ids = data.get("file_ids", [])
    research_depth = data.get("research_depth", "quick")
    
    user_info = get_current_user_info()
    user_id = user_info["user_id"] if user_info else data.get("user_id", "default_user")

    logger.info(f"[CHAT] user_id={user_id} chat_id={chat_id}")

    if not message:
        return jsonify({"success": False, "error": "User message is empty."}), 400

    # Build prompt context if attached files exist
    context_prefix = ""
    attached_file_titles = []
    if file_ids:
        attached_texts = []
        for fid in file_ids:
            if fid in uploaded_documents:
                doc = uploaded_documents[fid]
                attached_file_titles.append(doc["filename"])
                attached_texts.append(f"--- Document: {doc['filename']} ---\n{doc['text'][:3000]}")
        if attached_texts:
            context_prefix = "\n\n[USER ATTACHED DOCUMENTS CONTEXT]:\n" + "\n\n".join(attached_texts) + "\n\n[USER QUESTION]: "

    augmented_query = context_prefix + message if context_prefix else message

    try:
        result = pipeline_instance.run_pipeline(augmented_query, user_id=user_id, chat_id=chat_id, research_depth=research_depth)

        if result.get("success"):
            # Format sources nicely for SourceIQ UI
            raw_sources = result.get("sources", [])
            formatted_sources = []

            for idx, s in enumerate(raw_sources, 1):
                url = s.get("url", "")
                title = s.get("title") or f"Source {idx}"
                domain = url.split("/")[2] if "://" in url else "Web Document"
                formatted_sources.append({
                    "title": title,
                    "meta": f"Web Source · {domain}",
                    "snippet": f"Retrieved web source context from {title} ({url}).",
                    "url": url
                })

            # Include attached document sources if any
            for fname in attached_file_titles:
                formatted_sources.append({
                    "title": fname,
                    "meta": "User Uploaded Document",
                    "snippet": f"Content extracted directly from attached file '{fname}'.",
                    "url": "#"
                })

            result["sources"] = formatted_sources
            result["chat_id"] = chat_id

            # Auto save message to chat session if chat_id provided
            if chat_id:
                chats = load_chats()
                found = False
                for c in chats:
                    if c["id"] == chat_id:
                        found = True
                        if not c.get("messages"):
                            c["messages"] = []
                        c["messages"].append({"role": "user", "text": message})
                        c["messages"].append({
                            "role": "assistant",
                            "text": result["answer"],
                            "sources": formatted_sources,
                            "research_depth": result.get("research_depth", "quick"),
                            "sources_analyzed": result.get("sources_analyzed", 0),
                            "sources_used": result.get("sources_used", len(formatted_sources)),
                            "cache_mode": result.get("cache_mode")
                        })
                        if c.get("title") == "New chat" or not c.get("title"):
                            c["title"] = message[:46] + ("…" if len(message) > 46 else "")
                        break
                if not found:
                    chats.insert(0, {
                        "id": chat_id,
                        "title": message[:46] + ("…" if len(message) > 46 else ""),
                        "active": True,
                        "messages": [
                            {"role": "user", "text": message},
                            {
                                "role": "assistant",
                                "text": result["answer"],
                                "sources": formatted_sources,
                                "research_depth": result.get("research_depth", "quick"),
                                "sources_analyzed": result.get("sources_analyzed", 0),
                                "sources_used": result.get("sources_used", len(formatted_sources)),
                                "cache_mode": result.get("cache_mode")
                            }
                        ]
                    })
                save_chats(chats)

        return jsonify(result)
    except Exception as e:
        logger.error(f"Error handling /api/chat query: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/chats", methods=["GET", "POST"])
def chats_endpoint():
    chats = load_chats()
    if request.method == "GET":
        return jsonify({"success": True, "chats": chats})

    # POST: create new chat session
    new_id = f"chat_{uuid.uuid4().hex[:8]}"
    for c in chats:
        c["active"] = False
    
    new_chat = {
        "id": new_id,
        "title": "New chat",
        "active": True,
        "messages": []
    }
    chats.insert(0, new_chat)
    save_chats(chats)
    return jsonify({"success": True, "chat": new_chat, "chats": chats})

@app.route("/api/chats/<chat_id>", methods=["GET", "DELETE"])
def chat_detail(chat_id):
    chats = load_chats()
    if request.method == "DELETE":
        chats = [c for c in chats if c["id"] != chat_id]
        if not chats:
            # Always ensure at least one new chat exists
            new_id = f"chat_{uuid.uuid4().hex[:8]}"
            chats = [{ "id": new_id, "title": "New chat", "active": True, "messages": [] }]
        else:
            chats[0]["active"] = True
        save_chats(chats)
        return jsonify({"success": True, "chats": chats})

    # GET specific chat
    found = next((c for c in chats if c["id"] == chat_id), None)
    if found:
        for c in chats:
            c["active"] = (c["id"] == chat_id)
        save_chats(chats)
        return jsonify({"success": True, "chat": found})
    return jsonify({"success": False, "error": "Chat session not found"}), 404

@app.route("/api/clear", methods=["POST"])
def clear():
    pipeline_instance = get_pipeline()
    if not pipeline_instance:
        return jsonify({"success": False, "error": "Pipeline not initialized"}), 500
    try:
        pipeline_instance.chat_history.clear()
        uploaded_documents.clear()
        pipeline_instance.cache_manager.clear_all_cache()
        logger.info("Conversational chat history & uploaded files cleared.")
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=False)
