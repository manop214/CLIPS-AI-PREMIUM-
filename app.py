import os
import json
import uuid
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, send_from_directory
)
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
CLIP_DIR = ROOT / "clips"

UPLOAD_DIR.mkdir(exist_ok=True)
CLIP_DIR.mkdir(exist_ok=True)

app = Flask(__name__)

app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", "CHANGE-ME"),
    SQLALCHEMY_DATABASE_URI="sqlite:///" + str(ROOT / "clip_premium.db"),
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    MAX_CONTENT_LENGTH=2 * 1024 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE", "true").lower() == "true",
)

db = SQLAlchemy(app)
csrf = CSRFProtect(app)

ALLOWED = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}

# Evita varios procesamientos simultáneos.
processing_lock = threading.Lock()


# =========================
# MODELOS DE BASE DE DATOS
# =========================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)


class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_name = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(30), default="queued")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    error = db.Column(db.Text)


class Clip(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey("video.id"), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    title = db.Column(db.String(180), nullable=False)
    reason = db.Column(db.Text)
    start_seconds = db.Column(db.Float, nullable=False)
    end_seconds = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    video = db.relationship("Video", backref="clips")


# =========================
# AUTENTICACIÓN
# =========================

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper


def init_db():
    with app.app_context():
        db.create_all()

        username = os.environ.get("ADMIN_USERNAME", "owner")
        password = os.environ.get("ADMIN_PASSWORD")

        if password and not User.query.filter_by(username=username).first():
            db.session.add(
                User(
                    username=username,
                    password_hash=generate_password_hash(password)
                )
            )
            db.session.commit()


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password_hash, password):
            session.clear()
            session["user_id"] = user.id
            session.permanent = True
            return redirect(url_for("dashboard"))

        flash("Usuario o contraseña incorrectos.", "error")

    return render_template("login.html")


@app.post("/logout")
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))


# =========================
# PANEL PRINCIPAL
# =========================

@app.get("/")
@login_required
def dashboard():
    videos = Video.query.order_by(Video.created_at.desc()).all()
    clips = Clip.query.order_by(Clip.created_at.desc()).limit(100).all()

    return render_template(
        "dashboard.html",
        videos=videos,
        clips=clips
    )


# =========================
# SUBIR VÍDEO
# =========================

@app.post("/upload")
@login_required
def upload():
    f = request.files.get("video")

    if not f or not f.filename:
        flash("Selecciona un video.", "error")
        return redirect(url_for("dashboard"))

    original = secure_filename(f.filename)
    ext = Path(original).suffix.lower()

    if ext not in ALLOWED:
        flash("Formato no admitido.", "error")
        return redirect(url_for("dashboard"))

    stored = f"{uuid.uuid4().hex}{ext}"
    f.save(UPLOAD_DIR / stored)

    video = Video(
        filename=stored,
        original_name=original,
        status="queued"
    )

    db.session.add(video)
    db.session.commit()

    threading.Thread(
        target=process_video_safe,
        args=(video.id,),
        daemon=True
    ).start()

    flash("Video añadido. El análisis comenzará automáticamente.", "success")
    return redirect(url_for("dashboard"))


# =========================
# RECIBIR ENLACE
# =========================

@app.post("/add-video-link")
@login_required
def add_video_link():
    video_url = request.form.get("video_url", "").strip()

    if not video_url:
        flash("Pega un enlace de vídeo válido.", "error")
        return redirect(url_for("dashboard"))

    if not video_url.startswith(("https://", "http://")):
        flash("El enlace no es válido.", "error")
        return redirect(url_for("dashboard"))

    flash(
        "Enlace recibido. Para generar clips, sube un archivo "
        "de vídeo que tengas autorización para procesar.",
        "error"
    )

    return redirect(url_for("dashboard"))


# =========================
# REPROCESAR VÍDEO
# =========================

@app.post("/process/<int:video_id>")
@login_required
def process_one(video_id):
    threading.Thread(
        target=process_video_safe,
        args=(video_id,),
        daemon=True
    ).start()

    flash("Análisis iniciado.", "success")
    return redirect(url_for("dashboard"))


# =========================
# DESCARGAR CLIPS
# =========================

@app.get("/clips/<path:filename>")
@login_required
def get_clip(filename):
    return send_from_directory(
        CLIP_DIR,
        filename,
        as_attachment=True
    )


# =========================
# HERRAMIENTAS DE VÍDEO
# =========================

def run_cmd(args, timeout=3600):
    p = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout
    )

    if p.returncode != 0:
        raise RuntimeError(
            (p.stderr or "Error de FFmpeg")[-1800:]
        )

    return p.stdout


def video_duration(path):
    out = run_cmd([
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ], timeout=120)

    return float(out.strip())


# =========================
# INTELIGENCIA ARTIFICIAL
# =========================

def get_highlights(transcript, duration):
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("Falta OPENAI_API_KEY.")

    client = OpenAI(api_key=api_key)

    prompt = f"""
Analiza esta transcripción con marcas de tiempo.

Selecciona momentos interesantes, entretenidos, sorprendentes
o que tengan sentido por sí solos para clips de redes sociales.

Devuelve SOLO JSON:
{{
  "clips": [
    {{
      "start": 12.5,
      "end": 48.0,
      "title": "Título",
      "reason": "Motivo"
    }}
  ]
}}

Reglas:
- Los tiempos deben estar entre 0 y {duration:.1f}.
- Cada clip debe durar entre 8 y 90 segundos.
- No inventes contenido.
- Evita clips repetidos o solapados.
- Prioriza momentos que se entiendan sin mucho contexto.
- No impongas un máximo diario de clips.

TRANSCRIPCIÓN:
{transcript[:90000]}
"""

    response = client.chat.completions.create(
        model=os.environ.get("AI_TEXT_MODEL", "gpt-4o-mini"),
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "Eres un editor profesional de videos. Responde JSON válido."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    data = json.loads(response.choices[0].message.content)
    return data.get("clips", [])


# =========================
# PROCESAMIENTO DE VÍDEOS
# =========================

def process_video_safe(video_id):
    with processing_lock:
        try:
            with app.app_context():
                video = db.session.get(Video, video_id)

                if not video or video.status == "done":
                    return

                video.status = "processing"
                video.error = None
                db.session.commit()

                source = UPLOAD_DIR / video.filename
                audio = UPLOAD_DIR / f"{video_id}_audio.mp3"

                try:
                    if not source.exists():
                        raise RuntimeError("No se encontró el archivo de video.")

                    duration = video_duration(source)

                    # Extraer audio para transcripción.
                    run_cmd([
                        "ffmpeg", "-y",
                        "-i", str(source),
                        "-vn",
                        "-ac", "1",
                        "-ar", "16000",
                        "-b:a", "64k",
                        str(audio)
                    ], timeout=3600)

                    client = OpenAI(
                        api_key=os.environ["OPENAI_API_KEY"]
                    )

                    # Transcribir audio.
                    with open(audio, "rb") as af:
                        transcription = client.audio.transcriptions.create(
                            model=os.environ.get(
                                "AI_TRANSCRIBE_MODEL",
                                "gpt-4o-mini-transcribe"
                            ),
                            file=af,
                            response_format="verbose_json",
                            timestamp_granularities=["segment"]
                        )

                    segments = getattr(transcription, "segments", None) or []

                    transcript = "\n".join(
                        f"[{float(s.start):.1f}-{float(s.end):.1f}] {s.text}"
                        for s in segments
                    )

                    if not transcript:
                        transcript = getattr(transcription, "text", "")

                    moments = get_highlights(transcript, duration)

                    made = 0

                    # Generar clips sin límite diario.
                    for i, m in enumerate(moments):
                        start = max(0.0, float(m.get("start", 0)))
                        end = min(duration, float(m.get("end", 0)))

                        if end <= start or end - start < 8:
                            continue

                        outname = (
                            f"clip_{video.id}_{uuid.uuid4().hex[:8]}.mp4"
                        )

                        outpath = CLIP_DIR / outname

                        run_cmd([
                            "ffmpeg", "-y",
                            "-ss", str(start),
                            "-i", str(source),
                            "-t", str(end - start),
                            "-c:v", "libx264",
                            "-preset", "veryfast",
                            "-crf", "23",
                            "-c:a", "aac",
                            "-movflags", "+faststart",
                            str(outpath)
                        ], timeout=1800)

                        db.session.add(
                            Clip(
                                video_id=video.id,
                                filename=outname,
                                title=str(
                                    m.get("title", f"Momento destacado {i+1}")
                                )[:180],
                                reason=str(m.get("reason", ""))[:1000],
                                start_seconds=start,
                                end_seconds=end
                            )
                        )

                        db.session.commit()
                        made += 1

                    video.status = "done"
                    db.session.commit()

                except Exception as e:
                    video.status = "error"
                    video.error = str(e)[:3000]
                    db.session.commit()

                finally:
                    if audio.exists():
                        audio.unlink(missing_ok=True)

        except Exception as e:
            app.logger.exception("Error procesando video %s", video_id)


# =========================
# INICIALIZACIÓN
# =========================

init_db()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
    
