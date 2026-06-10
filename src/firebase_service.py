import os
from datetime import datetime, timezone

try:
    import pyrebase
except ImportError:  # pragma: no cover - handled gracefully at runtime
    pyrebase = None


def _load_env_file(env_path):
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def _database_path(*parts):
    return "/".join(part.strip("/") for part in parts if part)


def _slugify(value):
    cleaned = []
    for character in value.lower():
        if character.isalnum():
            cleaned.append(character)
        elif cleaned and cleaned[-1] != "_":
            cleaned.append("_")

    slug = "".join(cleaned).strip("_")
    return slug or "unknown"


class FirebaseService:
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self._firebase = None
        self._db = None
        self._auth_token = None
        self.is_ready = False

        _load_env_file(os.path.join(project_dir, ".env"))
        self._initialize()

    def _initialize(self):
        if pyrebase is None:
            return

        config = {
            "apiKey": os.getenv("FIREBASE_API_KEY", ""),
            "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", ""),
            "databaseURL": os.getenv("FIREBASE_DATABASE_URL", ""),
            "projectId": os.getenv("FIREBASE_PROJECT_ID", ""),
            "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET", ""),
            "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID", ""),
            "appId": os.getenv("FIREBASE_APP_ID", ""),
            "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID", ""),
        }

        required_keys = ("apiKey", "authDomain", "databaseURL", "projectId")
        if not all(config[key] for key in required_keys):
            return

        self._firebase = pyrebase.initialize_app(config)
        self._db = self._firebase.database()
        self._authenticate_if_possible()
        self.is_ready = True

    def _authenticate_if_possible(self):
        email = os.getenv("FIREBASE_AUTH_EMAIL", "")
        password = os.getenv("FIREBASE_AUTH_PASSWORD", "")
        if not email or not password:
            return

        try:
            auth = self._firebase.auth()
            user = auth.sign_in_with_email_and_password(email, password)
            self._auth_token = user.get("idToken")
        except Exception:
            self._auth_token = None

    def record_access_event(self, member_name, confidence, granted, member_id=None):
        if not self.is_ready:
            return False

        event_payload = {
            "member_name": member_name,
            "member_id": member_id,
            "confidence": confidence,
            "granted": granted,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "AuraGym",
        }

        member_key = _slugify(member_name)
        member_payload = {
            "name": member_name,
            "member_id": member_id,
            "last_confidence": confidence,
            "last_granted": granted,
            "last_seen": event_payload["timestamp"],
        }

        try:
            self._db.child(_database_path("access_logs")).push(event_payload, self._auth_token)
            self._db.child(_database_path("members", member_key)).set(member_payload, self._auth_token)
            return True
        except Exception:
            return False