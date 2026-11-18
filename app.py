import os
import uuid
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from functools import wraps

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, send_from_directory
)
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from faster_whisper import WhisperModel

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
processing_lock = threading.Lock()

whisper_model = None


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


@app.get("/clips/<path:filename>")
@login_required
def get_clip(filename):
    return send_from_directory(CLIP_DIR, filename, as_attachment=True)


def run_cmd(args, timeout=3600):
    p = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout
    )

    if p.returncode != 0:
        raise RuntimeError((p.stderr or "Error de FFmpeg")[-1800:])

    return p.stdout


def video_duration(path):
    out = run_cmd([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ], timeout=120)

    return float(out.strip())


def get_whisper():
    global whisper_model

    if whisper_model is None:
        whisper_model = WhisperModel(
            os.environ.get("WHISPER_MODEL", "base"),
            device="cpu",
            compute_type="int8"
        )

    return whisper_model


def select_moments(segments, duration):
    """
    Selección básica local, sin API de pago.
    Detecta segmentos con contenido suficiente y agrupa
    fragmentos cercanos para crear clips.
    """

    moments = []

    for s in segments:
        start = float(s.start)
        end = float(s.end)
        text = (s.text or "").strip()

        if not text:
            continue

        # Amplía un poco el contexto del segmento.
        start = max(0, start - 4)
        end = min(duration, end + 8)

        if end - start < 8:
            end = min(duration, start + 8)

        if end <= start:
            continue

        moments.append({
            "start": start,
            "end": end,
            "title": text[:80] or "Momento destacado",
            "reason": "Fragmento detectado en la transcripción."
        })

    # Quitar fragmentos demasiado próximos o solapados.
    selected = []

    for m in moments:
        if selected and m["start"] <= selected[-1]["end"]:
            selected[-1]["end"] = min(
                duration,
                max(selected[-1]["end"], m["end"])
            )
            continue

        selected.append(m)

    # Limitar la duración de los clips.
    final = []

    for m in selected:
        start = max(0, m["start"])
        end = min(duration, m["end"])

        if end - start > 90:
            end = start + 90

        if end - start >= 8:
            final.append({
                "start": start,
                "end": end,
                "title": m["title"],
                "reason": m["reason"]
            })

    return final


def process_video_safe(video_id):
    with processing_lock:
        with app.app_context():
            video = db.session.get(Video, video_id)

            if not video or video.status == "done":
                return

            video.status = "processing"
            video.error = None
            db.session.commit()

            source = UPLOAD_DIR / video.filename
            audio = UPLOAD_DIR / f"{video_id}_audio.wav"

            try:
                if not source.exists():
                    raise RuntimeError("No se encontró el archivo de video.")

                duration = video_duration(source)

                # Audio optimizado para transcripción local.
                run_cmd([
                    "ffmpeg", "-y",
                    "-i", str(source),
                    "-vn",
                    "-ac", "1",
                    "-ar", "16000",
                    str(audio)
                ], timeout=3600)

                model = get_whisper()

                segments, info = model.transcribe(
                    str(audio),
                    vad_filter=True
                )

                segments = list(segments)

                moments = select_moments(segments, duration)

                made = 0

                for i, m in enumerate(moments):
                    start = float(m["start"])
                    end = float(m["end"])

                    if end <= start or end - start < 8:
                        continue

                    outname = f"clip_{video.id}_{uuid.uuid4().hex[:8]}.mp4"
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
                            title=m["title"][:180],
                            reason=m["reason"][:1000],
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


init_db()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
