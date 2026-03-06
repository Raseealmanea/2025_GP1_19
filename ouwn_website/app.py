from dotenv import load_dotenv
load_dotenv()
from typing import Dict, Any

import os, json, re, uuid
from datetime import datetime, date, timezone

from flask import Flask, render_template, jsonify, request, session, redirect, url_for, flash
from firebase.Initialization import db

import whisper
import hashlib
 

# ----------------------------
# ML imports (ICD model)
# ----------------------------
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, RobertaModel

DEVICE = "cpu"  # change to "cuda" if you have GPU + proper torch build


# ---- EXACT LabelAttention (same as repo) ----
class LabelAttention(nn.Module):
    def __init__(self, input_size: int, projection_size: int, num_classes: int):
        super().__init__()
        self.first_linear = nn.Linear(input_size, projection_size, bias=False)
        self.second_linear = nn.Linear(projection_size, num_classes, bias=False)
        self.third_linear = nn.Linear(input_size, num_classes)
        self._init_weights(mean=0.0, std=0.03)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.tanh(self.first_linear(x))
        att_weights = self.second_linear(weights)
        att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1, 2)
        weighted_output = att_weights @ x
        return (
            self.third_linear.weight.mul(weighted_output)
            .sum(dim=2)
            .add(self.third_linear.bias)
        )

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        torch.nn.init.normal_(self.first_linear.weight, mean, std)
        torch.nn.init.normal_(self.second_linear.weight, mean, std)
        torch.nn.init.normal_(self.third_linear.weight, mean, std)


# ---- EXACT PLMICD forward (same as repo) ----
class PLMICD(nn.Module):
    def __init__(self, num_classes: int, model_path: str):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            model_path, num_labels=num_classes, finetuning_task=None
        )

        # ✅ safer load
        self.roberta = RobertaModel.from_pretrained(
            model_path, config=self.config, add_pooling_layer=False
        )

        self.attention = LabelAttention(
            input_size=self.config.hidden_size,
            projection_size=self.config.hidden_size,
            num_classes=num_classes,
        )

    def forward(self, input_ids=None, attention_mask=None):
        batch_size, num_chunks, chunk_size = input_ids.size()

        outputs = self.roberta(
            input_ids.view(-1, chunk_size),
            attention_mask=attention_mask.view(-1, chunk_size) if attention_mask is not None else None,
            return_dict=False,
        )

        hidden_output = outputs[0].view(batch_size, num_chunks * chunk_size, -1)
        logits = self.attention(hidden_output)
        return logits

import re

def preprocess_note_text_exact(text: str) -> str:
    """
    Approximation of JoakimEdin prepare_mimiciv.py preprocessing:
    - lower=True
    - remove_digits=True
    - mullenbach-style cleanup (keep letters, spaces; normalize separators)
    """
    if text is None:
        return ""

    s = str(text).lower()

    # ✅ IMPORTANT (matches training): remove digits
    s = re.sub(r"\d+", " ", s)

    # mullenbach-ish: normalize common separators to spaces
    s = s.replace("\n", " ").replace("\t", " ")

    # remove weird characters but keep letters and basic punctuation if you want:
    # (this is safer than deleting everything not A-Za-z0-9)
    s = re.sub(r"[^a-z\s\.,;:\-\(\)\/]+", " ", s)

    # collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_state(sd: dict) -> dict:
    return {k.replace("module.", ""): v for k, v in sd.items()}


# lazy-load globals
model = None
tokenizer = None
BEST_THRESHOLD = None
index2target = None
NUM_LABELS = None
_model_loaded = False


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _extract_threshold_from_ckpt(ckpt: Dict[str, Any]) -> float:
    """
    Tries multiple key names for threshold.
    If missing/unusable -> returns default 0.5 (NO crash).
    """
    thr_obj = ckpt.get("best_threshold", ckpt.get("threshold", ckpt.get("BEST_THRESHOLD", None)))

    if thr_obj is None:
        print("⚠️ No threshold found in checkpoint. Using default 0.5. Keys:", list(ckpt.keys()))
        return 0.5

    if isinstance(thr_obj, dict):
        if "all" in thr_obj:
            thr_obj = thr_obj["all"]
        else:
            # pick first convertible value
            for v in thr_obj.values():
                try:
                    thr_obj = float(v)
                    break
                except Exception:
                    continue

    try:
        return float(thr_obj)
    except Exception:
        print("⚠️ Threshold found but not float-castable. Using default 0.5. Value:", thr_obj)
        return 0.5

def ensure_model_loaded(app: Flask) -> None:
    """
    Lazy-load ICD model from:
      app.root_path/models/best_model.pt
      app.root_path/models/target2index.json
      app.root_path/models/RoBERTa-base-PM-M3-Voc-hf/
    """
    global model, tokenizer, BEST_THRESHOLD, index2target, NUM_LABELS, _model_loaded

    if _model_loaded:
        return

    models_dir = os.path.join(app.root_path, "models")
    MODEL_PATH = os.path.join(models_dir, "RoBERTa-base-PM-M3-Voc-hf")
    CKPT_PATH  = os.path.join(models_dir, "best_model.pt")
    T2I_PATH   = os.path.join(models_dir, "target2index.json")

    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Missing models directory: {models_dir}")
    if not os.path.exists(T2I_PATH):
        raise FileNotFoundError(f"Missing target2index.json at: {T2I_PATH}")
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Missing best_model.pt at: {CKPT_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model folder at: {MODEL_PATH}")

    print("✅ Loading checkpoint from:", CKPT_PATH)
    print("✅ best_model.pt sha256:", _file_sha256(CKPT_PATH))

    with open(T2I_PATH, "r", encoding="utf-8") as f:
        target2index = json.load(f)

    index2target = {int(v): str(k) for k, v in target2index.items()}
    NUM_LABELS = len(index2target)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    print("ℹ️ CKPT keys:", list(ckpt.keys()))

    BEST_THRESHOLD = _extract_threshold_from_ckpt(ckpt)
    print("✅ Threshold used:", BEST_THRESHOLD)

    if "model" in ckpt:
        state = ckpt["model"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        raise KeyError(f"Checkpoint missing 'model'/'state_dict'. Keys: {list(ckpt.keys())}")

    state = normalize_state(state)

    tokenizer_local = AutoTokenizer.from_pretrained(MODEL_PATH)
    model_obj = PLMICD(num_classes=NUM_LABELS, model_path=MODEL_PATH).to(DEVICE)

    missing, unexpected = model_obj.load_state_dict(state, strict=False)
    print("ℹ️ load_state_dict missing keys:", len(missing))
    print("ℹ️ load_state_dict unexpected keys:", len(unexpected))

    model_obj.eval()

    tokenizer = tokenizer_local
    model = model_obj
    _model_loaded = True

    print("✅ ICD model loaded successfully FROM best_model.pt")


# ---------------------------------------------------------
# Create Flask App
# ---------------------------------------------------------
# Create Flask App
def create_app():
    app = Flask(__name__)
    # secret key for sessions
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "fallback-secret-key")
    app.config["PROPAGATE_EXCEPTIONS"] = True # Allow Flask to show detailed exceptions
        # --- Whisper STT setup ---
    app.config["WHISPER_MODEL"] = os.environ.get("WHISPER_MODEL", "base")
    app.config["AUDIO_UPLOAD_DIR"] = os.path.join(app.root_path, "uploads_audio")
    os.makedirs(app.config["AUDIO_UPLOAD_DIR"], exist_ok=True)

    # Load whisper model once (fast for later requests)
    app.whisper_model = whisper.load_model(app.config["WHISPER_MODEL"])



     # loading all blueprints (auth + reset)
    from routes.Authentication import auth_bp
    app.register_blueprint(auth_bp)

    from routes.auth_reset import reset_bp
    app.register_blueprint(reset_bp)

    # Load ICD JSON data
    ICD_FILE = os.path.join(app.root_path, "static", "icd_data.json")
    if os.path.exists(ICD_FILE):
        with open(ICD_FILE, "r", encoding="utf-8") as f:
            app.icd_data = json.load(f) # Store JSON in memory
    else:
        print("⚠️ icd_data.json missing in /static")
        app.icd_data = []

    # homePage
    @app.route("/")
    def home():
        return render_template("homePage.html")

    def get_filtered_patients(search_query="", yob_query="", icd_query="", include_meta=False):
        patients = []
        docs = db.collection("Patient").stream()

        for doc in docs:
            data = doc.to_dict()

            dob = data.get("DOB")
            age = None
            dob_date = None

            if dob:
                try:
                    if isinstance(dob, datetime):
                        dob_date = dob.date()
                    else:
                        dob_date = datetime.strptime(str(dob), "%Y-%m-%d").date()

                    age = date.today().year - dob_date.year
                    if (date.today().month, date.today().day) < (dob_date.month, dob_date.day):
                        age -= 1
                except:
                    age = None
                    dob_date = None

            # -------- NAME / ID FILTER --------
            name_match = True
            if search_query:
                full_name = data.get("FullName", "").lower()
                patient_id = doc.id.lower()
                name_parts = full_name.split()
                name_match = any(part.startswith(search_query) for part in name_parts) or patient_id.startswith(search_query)

            # -------- YEAR OF BIRTH FILTER (DOB year prefix) --------
            yob_match = True
            if yob_query:
                q = str(yob_query).strip()

                # allow only digits (extra safety)
                if not q.isdigit():
                    yob_match = False
                else:
                    if dob_date is None:
                        yob_match = False
                    else:
                        year_str = str(dob_date.year)   # e.g., "1888", "1999", "2004"
                        yob_match = year_str.startswith(q)

            # -------- ICD PREFIX FILTER --------
            icd_match = True
            icd_date = None

            if icd_query or include_meta:
                # Only do the expensive notes scan if needed (icd search OR icd_date sorting)
                icd_match = False if icd_query else True

                notes = db.collection("Patient").document(doc.id).collection("MedicalNote").stream()
                for n in notes:
                    icds = n.reference.collection("ICDcode").stream()
                    for icd_doc in icds:
                        d = icd_doc.to_dict()

                        # For filtering by ICD prefix
                        if icd_query:
                            for code in d.get("Adjusted", []):
                                if str(code).upper().startswith(icd_query):
                                    icd_match = True
                                    break

                        # For sorting by earliest ICD date
                        if include_meta:
                            adjusted_at = d.get("AdjustedAt")
                            if adjusted_at:
                                if isinstance(adjusted_at, datetime):
                                    icd_date = min(icd_date, adjusted_at) if icd_date else adjusted_at
                                else:
                                    try:
                                        dt = datetime.strptime(adjusted_at, "%Y-%m-%d %H:%M:%S")
                                        icd_date = min(icd_date, dt) if icd_date else dt
                                    except:
                                        pass

                        if icd_query and icd_match and not include_meta:
                            break
                    if icd_query and icd_match and not include_meta:
                        break

            if name_match and yob_match and icd_match:
                patient_obj = {
                    "ID": doc.id,
                    "FullName": data.get("FullName", "Unknown"),
                    "Age": age
                }

                if include_meta:
                    patient_obj["DOB_Date"] = dob_date
                    patient_obj["ICDDate"] = icd_date

                patients.append(patient_obj)

        return patients

    # DASHBOARD
    @app.route("/dashboard")
    def dashboard():
        if 'user_id' not in session:
            return redirect(url_for('Authentication.login'))

        search_query = request.args.get("search", "").strip().lower()
        yob_query = request.args.get("yob", "").strip() or request.args.get("age", "").strip()
        icd_query = request.args.get("icd", "").strip().upper()

        try:
            patients = get_filtered_patients(
                search_query=search_query,
                yob_query=yob_query,
                icd_query=icd_query,
                include_meta=False   
            )
        except Exception as e:
            patients = []
            flash(f"Error fetching patients: {e}", "danger")

        # message after adding patient
        msg_key = request.args.get('msg', '')
        msg_text = {
            "patient_added": "Patient added successfully!",
            "added": "Patient added successfully!"
        }.get(msg_key, "")


        return render_template("dashboard.html", patients=patients, msg_text=msg_text)

    @app.route("/api/patients")
    def api_patients():
        if 'user_id' not in session:
            return jsonify([])

        search_query = request.args.get("search", "").strip().lower()
        yob_query = request.args.get("yob", "").strip() or request.args.get("age", "").strip()
        icd_query = request.args.get("icd", "").strip().upper()
        sort = request.args.get("sort", "").strip()

        # Need meta for these sorts
        include_meta = sort in ("age_young", "age_old", "icd_date")

        patients = get_filtered_patients(
            search_query=search_query,
            yob_query=yob_query,
            icd_query=icd_query,
            include_meta=include_meta
        )

        # ---- SORTING ----
        if sort == "name_asc":
            patients.sort(key=lambda x: (x.get("FullName") or "").lower())

        elif sort == "name_desc":
            patients.sort(key=lambda x: (x.get("FullName") or "").lower(), reverse=True)

        elif sort == "age_young":
            patients.sort(key=lambda x: (
                x.get("Age") is None,
                x.get("Age") if x.get("Age") is not None else 0,
                x.get("DOB_Date") is None,
                -(x["DOB_Date"].toordinal() if x.get("DOB_Date") else 0)
            ))

        elif sort == "age_old":
            patients.sort(key=lambda x: (
                x.get("Age") is None,
                -(x.get("Age") if x.get("Age") is not None else 0),
                x.get("DOB_Date") is None,
                (x["DOB_Date"].toordinal() if x.get("DOB_Date") else 0)
            ))

        elif sort == "icd_date":
            def normalize(dt):
                if dt is None:
                    return datetime.max
                if getattr(dt, "tzinfo", None) is not None:
                    return dt.astimezone(timezone.utc).replace(tzinfo=None)
                return dt

            patients.sort(key=lambda x: normalize(x.get("ICDDate")))

        # IMPORTANT: remove non-JSON-safe fields before returning
        for p in patients:
            p.pop("DOB_Date", None)
            p.pop("ICDDate", None)

        return jsonify(patients)

    # ADD PATIENT
    @app.route("/add_patient", methods=["GET", "POST"])
    def add_patient():
        if 'user_id' not in session:
            return redirect(url_for('Authentication.login'))

        errors = [] # Collect validation errors
        # form data
        if request.method == "POST":
            pid = request.form.get("ID", "").strip()
            name = request.form.get("full_name", "").strip()
            dob = request.form.get("dob", "").strip()
            gender = request.form.get("gender", "").strip()
            phone = request.form.get("phone", "").strip()
            email = request.form.get("email", "").strip()
            address = request.form.get("address", "").strip()
            blood = request.form.get("blood_type", "").strip()

            # Required fields check
            if not all([pid, name, dob, gender, phone, email, address, blood]):
                errors.append("All fields are required.")

            # ID format check (National ID must be 10 digits)
            if pid and not re.fullmatch(r'\d{10}', pid):
                errors.append("ID must be exactly 10 digits.")

            # Phone format check
            if phone and not re.fullmatch(r'^05\d{8}$', phone):
                errors.append("Phone must start with 05 and be 10 digits.")

            # Email format check
            if email and not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
                errors.append("Invalid email format.")
            
            # dob validation
            if dob:
                try:
                    dob_date = datetime.strptime(dob, "%Y-%m-%d").date()
                    if dob_date > date.today():
                        errors.append("Date of Birth cannot be in the future.")

                    # 2) Age must be <= 130
                    age = date.today().year - dob_date.year
                    if (date.today().month, date.today().day) < (dob_date.month, dob_date.day):
                        age -= 1

                    if age > 130:
                        errors.append("Age cannot exceed 130 years.")
                except ValueError:
                    errors.append("Invalid date format.")
            # check if patient exists already
            if not errors:
                if db.collection("Patient").document(pid).get().exists:
                    errors.append("Patient already exists with this ID.")
            # save to Firestore if everything is good
            if not errors:
                db.collection("Patient").document(pid).set({
                    "ID": pid,                     # PATIENT ID = National ID
                    "FullName": name,
                    "DOB": dob,
                    "Gender": gender,
                    "Phone": phone,
                    "Email": email,
                    "Address": address,
                    "BloodType": blood,
                    "CreatedBy": session['user_id']    #  Doctor who added the patient
                })
                return redirect(url_for("dashboard", msg="added"))

        return render_template("add_patient.html", errors=errors)



    # MEDICAL NOTES + ICD (save)
    @app.route("/MedicalNotes", methods=["GET", "POST"])
    def add_note():
        if "user_id" not in session:
            return redirect(url_for("Authentication.login"))

        if request.method == "GET":
            pid = request.args.get("pid", "").strip()
            patient_name = ""

            if pid:
                doc = db.collection("Patient").document(pid).get()
                if doc.exists:
                    patient_name = (doc.to_dict() or {}).get("FullName", "")

            return render_template(
                "MedicalNotes.html",
                prefilled_pid=pid,
                prefilled_name=patient_name,
                note_text="",
                selected_icd_codes=[]
            )

        try:
            data = request.get_json(silent=True) or request.form or {}
            pid = (data.get("pid") or "").strip()
            note_text = (data.get("note_text") or "").strip()
            icd_codes = data.get("icd_codes", [])
            predicted_codes = data.get("predicted_codes", [])

            if not pid or not note_text or not icd_codes:
                return jsonify({"status": "error", "message": "Missing fields"}), 400

            patient_ref = db.collection("Patient").document(pid)

            note_id = "note_id_" + uuid.uuid4().hex[:8]
            note_ref = patient_ref.collection("MedicalNote").document(note_id)

            note_ref.set({
                "NoteID": note_id,
                "Note": note_text,
                "CreatedDate": datetime.now(),
                "CreatedBy": session.get("user_id")
            })

            icd_id = "icdcode_id_" + uuid.uuid4().hex[:8]
            icd_ref = note_ref.collection("ICDcode").document(icd_id)

            icd_ref.set({
                "ICD_ID": icd_id,
                "Adjusted": [c["Code"] for c in icd_codes],
                "Predicted": predicted_codes,
                "AdjustedBy": session.get("user_id"),
                "AdjustedAt": datetime.now()
            })

            return jsonify({"status": "success", "redirect": url_for("dashboard")})

        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
 # AJAX CHECK ID
    @app.route("/check_id")
    def check_id():
        if 'user_id' not in session:
            return jsonify({"error": "Unauthorized"}), 401

        v = request.args.get("v", "").strip()
        exists = db.collection("Patient").document(v).get().exists if v else False
        return jsonify({"exists": exists})

    
    # extract categories from JSON 
    @app.route("/icd_categories")
    def icd_categories():
        if 'user_id' not in session:
            return jsonify({"error": "Unauthorized"}), 401

        categories = sorted({cat["Category"] for cat in app.icd_data})
        categories.insert(0, "All") # Add "All" option
        return jsonify(categories)


    # Return ICD codes by category
    @app.route("/icd_by_category/<path:category>")
    def icd_by_category(category):
        if 'user_id' not in session:
            return jsonify({"error": "Unauthorized"}), 401

        results = []

        if category.lower() == "all":
             # Return all codes
            for cat in app.icd_data:
                results.extend(cat.get("Codes", []))
        else:
            # Return only selected category
            for cat in app.icd_data:
                if cat["Category"].strip().lower() == category.strip().lower():
                    results = cat.get("Codes", [])
                    break

        return jsonify(results[:100])

    # Search ICD codes by term and code
    @app.route("/search_icd")
    def search_icd():
        if 'user_id' not in session:
            return jsonify({"error": "Unauthorized"}), 401

        term = request.args.get("term", "").lower()
        category = request.args.get("category", "").lower()

        if not term:
            return jsonify([])

        results = []
        # search inside ICD JSON
        for cat in app.icd_data:
            if category and category != "all" and cat["Category"].lower() != category:
                continue
            for code in cat["Codes"]:
                if (
                    term in code["Code"].lower()
                    or term in code["Description"].lower()
                    or term in cat["Category"].lower()   # ✅ search category
                ):
                    results.append({
                        "Code": code["Code"],
                        "Description": code["Description"],
                        "Category": cat["Category"]      # ✅ send category to frontend
                    })

        # remove duplicates
        unique = {item["Code"]: item for item in results}
        return jsonify(list(unique.values())[:30])

 
    # PROFILE
    @app.route("/profile", methods=["GET", "POST"])
    def profile():
        if 'user_id' not in session:
            return redirect(url_for('Authentication.login'))

        old_id = session['user_id']
        old_ref = db.collection('HealthCareP').document(old_id)
        doc = old_ref.get()

        # Load current user data
        current_user = doc.to_dict() if doc.exists else {"Name": "", "UserID": "", "Email": ""}

        # Update Profile
        if request.method == "POST" and request.form.get("action") == "update_profile":

            new_name = request.form.get("name", "").strip()
            new_email = request.form.get("email", "").strip()
            new_username = request.form.get("username", "").strip()

            try:

                # Email validation
                if not re.match(r"^[^@]+@[^@]+\.[^@]+$", new_email):
                    flash("Invalid email format.", "error")
                    return redirect(url_for("profile"))

                # Username validation
                if not re.fullmatch(r"^[A-Z][A-Za-z0-9._-]{2,31}$", new_username):
                    flash("Username must start with a CAPITAL letter and be 3–32 characters.", "error")
                    return redirect(url_for("profile"))
   
                # No changes → do nothing and show no message
                if (new_name == current_user["Name"] 
                    and new_email == current_user["Email"] 
                    and new_username.strip().lower() == old_id.strip().lower()):
                    return redirect(url_for("profile"))
                
                # Check if email belongs to another user
                email_query = db.collection("HealthCareP").where("Email", "==", new_email).stream()
                for docx in email_query:
                    if docx.id != old_id:  # email belongs to different user
                        return redirect(url_for("profile"))  # silently ignore update
                    
                if new_username.strip().lower() != old_id.strip().lower():
                    all_users = db.collection("HealthCareP").stream()
                    for u in all_users:
                        if u.id.strip().lower() == new_username.strip().lower():
                            return redirect(url_for("profile"))  # silent abort

    
                # updating without username change 
                if new_username.strip().lower() == old_id.strip().lower():
                    old_ref.update({
                        "Name": new_name,
                        "Email": new_email
                    })
                    flash("Profile updated successfully!", "success")
                    return redirect(url_for("profile"))

                # If username CHANGED , create new doc 
                clean_username = new_username.strip()
                new_ref = db.collection("HealthCareP").document(clean_username)


                # Copy data to new doc
                new_ref.set({
                    "Name": new_name,
                    "Email": new_email,
                    "UserID": clean_username,
                    "Password": current_user["Password"]
                })

                # Delete old document
                old_ref.delete()

                # Update session
                session['user_id'] = clean_username
                session['user_name'] = new_name
                session['user_email'] = new_email

                
                # fix all Firestore references that stored the old doctor ID
                old_doctor_id = old_id
                new_doctor_id = new_username

                # Update Patients.CreatedBy
                patients = db.collection("Patient").where("CreatedBy", "==", old_doctor_id).stream()
                for p in patients:
                    p.reference.update({"CreatedBy": new_doctor_id})

                # Update MedicalNotes.CreatedBy
                patients_all = db.collection("Patient").stream()
                for p in patients_all:
                    notes = p.reference.collection("MedicalNote").where("CreatedBy", "==", old_doctor_id).stream()
                    for n in notes:
                        n.reference.update({"CreatedBy": new_doctor_id})

                # Update ICDCode.AdjustedBy
                patients_all = db.collection("Patient").stream()
                for p in patients_all:
                    notes = p.reference.collection("MedicalNote").stream()
                    for n in notes:
                        icds = n.reference.collection("ICDcode").where("AdjustedBy", "==", old_doctor_id).stream()
                        for icd in icds:
                            icd.reference.update({"AdjustedBy": new_doctor_id})

                flash("Profile + all related records updated successfully!", "success")
                return redirect(url_for("profile"))

            except Exception as e:
                flash(str(e), "error")
                return redirect(url_for("profile"))

        return render_template("profile.html", user=current_user)

    @app.route("/check")
    def check_unique():
        if 'user_id' not in session:
            return jsonify({"ok": False, "valid": False, "exists": False})

        field = request.args.get("field", "")
        value = request.args.get("value", "").strip()
        current_user = session['user_id'].strip()

        # 1) Validate empty field
        if not field or not value:
            return jsonify({"ok": True, "valid": False, "exists": False})

        # 2) Username validation
        if field == "username":
            value_lower = value.lower()

            # Local validation rule (Capital letter, 3-32 chars)
            if not re.fullmatch(r"^[A-Z][A-Za-z0-9._-]{2,31}$", value):
                return jsonify({"ok": True, "valid": False, "exists": False})

            # Ignore user old username
            if value_lower == current_user.lower():
                return jsonify({"ok": True, "valid": True, "exists": False})

            # Check Firestore for duplicates
            all_users = db.collection("HealthCareP").stream()
            for u in all_users:
                if u.id.strip().lower() == value_lower:
                    return jsonify({"ok": True, "valid": True, "exists": True})

            return jsonify({"ok": True, "valid": True, "exists": False})


        # 3) Email validation
        if field == "email":
            if not re.match(r"^[^@]+@[^@]+\.[^@]+$", value):
                return jsonify({"ok": True, "valid": False, "exists": False})

            value_lower = value.lower()
            # Ignore user old email
            user_doc = None
            for u in db.collection("HealthCareP").stream():
                if u.id.lower() == current_user.lower():
                    user_doc = u
                    break
            # If same as your current email → OK
            if user_doc:
                if user_doc.to_dict().get("Email", "").strip().lower() == value_lower:
                    return jsonify({"ok": True, "valid": True, "exists": False})

            # Check duplicates
            email_query = db.collection("HealthCareP").where("Email", "==", value).stream()
            for doc in email_query:
                if doc.id != current_user:
                    return jsonify({"ok": True, "valid": True, "exists": True})

            return jsonify({"ok": True, "valid": True, "exists": False})

        # 4) Default fallback
        return jsonify({"ok": False, "valid": False, "exists": False})


    # LOGOUT
    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("home"))

    @app.route("/transcribe", methods=["POST"])
    def transcribe_audio():
        # Must be logged in
        if 'user_id' not in session:
            return jsonify({"error": "Unauthorized"}), 401

        if "audio" not in request.files:
            return jsonify({"error": "No audio file found. Key must be 'audio'."}), 400

        audio_file = request.files["audio"]
        if audio_file.filename == "":
            return jsonify({"error": "Empty filename."}), 400

        # Save temporarily
        ext = os.path.splitext(audio_file.filename)[1].lower() or ".webm"
        temp_name = f"{uuid.uuid4().hex}{ext}"
        temp_path = os.path.join(app.config["AUDIO_UPLOAD_DIR"], temp_name)
        audio_file.save(temp_path)

        try:
            # Auto-detect language (you can force language="en" or "ar" if you want)
            result = app.whisper_model.transcribe(temp_path)
            text = (result.get("text") or "").strip()
            language = result.get("language", "unknown")
            return jsonify({"text": text, "language": language})
        except Exception as e:
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass

    def delete_collection(coll_ref, batch_size=50):
        docs = coll_ref.limit(batch_size).stream()
        deleted = 0

        for doc in docs:
            doc.reference.delete()
            deleted += 1

        if deleted >= batch_size:
            return delete_collection(coll_ref, batch_size)
        return deleted


    def delete_patient_everything(patient_ref):
        """
        Deletes:
          Patient/{pid}
            MedicalNote/{noteId}
              ICDcode/{icdId}
        then deletes the patient doc itself.
        """
        notes_ref = patient_ref.collection("MedicalNote")
        notes = list(notes_ref.stream())

        for note_doc in notes:
            note_ref = notes_ref.document(note_doc.id)

            # delete ICD codes under this note
            icd_ref = note_ref.collection("ICDcode")
            delete_collection(icd_ref, batch_size=50)

            # delete the note document
            note_ref.delete()

        # delete patient document
        patient_ref.delete()


    @app.route("/delete_patient/<pid>", methods=["POST"])
    def delete_patient(pid):
        if 'user_id' not in session:
            return jsonify({"error": "Unauthorized"}), 401

        pid = (pid or "").strip()

        if not re.fullmatch(r"\d{10}", pid):
            return jsonify({"error": "Invalid patient ID."}), 400

        try:
            patient_ref = db.collection("Patient").document(pid)
            doc = patient_ref.get()

            if not doc.exists:
                return jsonify({"error": "Patient not found."}), 404

            delete_patient_everything(patient_ref)
            return jsonify({"status": "success"})

        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
        
    # ---------------------------
    # UPDATE PATIENT INFO ONLY
    # ---------------------------
    @app.route("/edit_patient/<pid>", methods=["GET", "POST"])
    def edit_patient(pid):
        if 'user_id' not in session:
            return redirect(url_for('Authentication.login'))

        pid = (pid or "").strip()
        if not re.fullmatch(r"\d{10}", pid):
            return redirect(url_for("dashboard"))

        patient_ref = db.collection("Patient").document(pid)
        patient_doc = patient_ref.get()
        if not patient_doc.exists:
            return redirect(url_for("dashboard"))

        pdata = patient_doc.to_dict() or {}
        errors = []

        # Default form values (GET)
        form = {
            "full_name": pdata.get("FullName", ""),
            "dob": pdata.get("DOB", ""),
            "gender": pdata.get("Gender", ""),
            "phone": pdata.get("Phone", ""),
            "email": pdata.get("Email", ""),
            "address": pdata.get("Address", ""),
            "blood_type": pdata.get("BloodType", ""),
        }

        if request.method == "POST":
            full_name = request.form.get("full_name", "").strip()
            dob = request.form.get("dob", "").strip()
            gender = request.form.get("gender", "").strip()
            phone = request.form.get("phone", "").strip()
            email = request.form.get("email", "").strip()
            address = request.form.get("address", "").strip()
            blood = request.form.get("blood_type", "").strip()

            form = {
                "full_name": full_name,
                "dob": dob,
                "gender": gender,
                "phone": phone,
                "email": email,
                "address": address,
                "blood_type": blood,
            }

            # ---- validations (keep yours) ----
            if not all([full_name, dob, gender, phone, email, address, blood]):
                errors.append("All fields are required.")

            if phone and not re.fullmatch(r'^05\d{8}$', phone):
                errors.append("Phone must start with 05 and be 10 digits.")

            if email and not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
                errors.append("Invalid email format.")

            if dob:
                try:
                    dob_date = datetime.strptime(dob, "%Y-%m-%d").date()
                    if dob_date > date.today():
                        errors.append("Date of Birth cannot be in the future.")

                    age = date.today().year - dob_date.year
                    if (date.today().month, date.today().day) < (dob_date.month, dob_date.day):
                        age -= 1
                    if age > 130:
                        errors.append("Age cannot exceed 130 years.")
                except ValueError:
                    errors.append("Invalid date format.")

            if errors:
                return render_template("edit_patient.html", pid=pid, form=form, errors=errors)

            # ---- detect changes (server-side truth) ----
            def norm(s):
                return (s or "").strip()

            changed = (
                norm(pdata.get("FullName")) != full_name or
                norm(pdata.get("DOB")) != dob or
                norm(pdata.get("Gender")) != gender or
                norm(pdata.get("Phone")) != phone or
                norm(pdata.get("Email")) != email or
                norm(pdata.get("Address")) != address or
                norm(pdata.get("BloodType")) != blood
            )

            if changed:
                patient_ref.update({
                    "FullName": full_name,
                    "DOB": dob,
                    "Gender": gender,
                    "Phone": phone,
                    "Email": email,
                    "Address": address,
                    "BloodType": blood,
                })
                return redirect(url_for("view_patient", pid=pid, saved="1"))

            # no changes
            return redirect(url_for("view_patient", pid=pid))

        return render_template("edit_patient.html", pid=pid, form=form, errors=errors)
    
    # ---------------------------
    # MEDICAL NOTES EDIT PAGE (LIST)
    # ---------------------------
    @app.route("/edit_medical_notes/<pid>", methods=["GET"])
    def edit_medical_notes(pid):
        if 'user_id' not in session:
            return redirect(url_for('Authentication.login'))

        pid = (pid or "").strip()
        if not re.fullmatch(r"\d{10}", pid):
            return redirect(url_for("dashboard"))

        patient_ref = db.collection("Patient").document(pid)
        patient_doc = patient_ref.get()
        if not patient_doc.exists:
            return redirect(url_for("dashboard"))

        pdata = patient_doc.to_dict() or {}
        patient_name = pdata.get("FullName", "")

        notes_out = []
        notes = patient_ref.collection("MedicalNote").order_by("CreatedDate", direction="DESCENDING").stream()
        for n in notes:
            nd = n.to_dict() or {}
            created = nd.get("CreatedDate")
            created_str = ""
            try:
                if isinstance(created, datetime):
                    created_str = created.strftime("%Y-%m-%d %H:%M")
                elif created:
                    created_str = str(created)
            except:
                created_str = ""

            icd_id = ""
            icd_codes = []
            try:
                icd_docs = list(n.reference.collection("ICDcode").limit(1).stream())
                if icd_docs:
                    icd_id = icd_docs[0].id
                    icd_data = icd_docs[0].to_dict() or {}
                    icd_codes = icd_data.get("Adjusted", []) or []
            except:
                pass

            icd_codes_str = ", ".join([str(x).strip().upper() for x in icd_codes if str(x).strip()])

            notes_out.append({
                "note_id": n.id,
                "note_text": nd.get("Note", ""),
                "created_date": created_str,
                "icd_id": icd_id,
                "icd_codes_str": icd_codes_str
            })

            

        return render_template("edit_medical_notes.html", pid=pid, patient_name=patient_name, notes=notes_out, errors=[])


    # ---------------------------
    # UPDATE ONE MEDICAL NOTE ONLY
    # ---------------------------
    @app.route("/edit_medical_notes/<pid>/<note_id>", methods=["POST"])
    def edit_medical_note(pid, note_id):
        if 'user_id' not in session:
            return redirect(url_for('Authentication.login'))

        pid = (pid or "").strip()
        note_id = (note_id or "").strip()

        if not re.fullmatch(r"\d{10}", pid) or not note_id:
            return redirect(url_for("dashboard"))

        patient_ref = db.collection("Patient").document(pid)
        patient_doc = patient_ref.get()
        if not patient_doc.exists:
            return redirect(url_for("dashboard"))

        note_text = request.form.get("note_text", "").strip()
        icd_id = (request.form.get("icd_id") or "").strip()
        raw_codes = request.form.get("icd_codes", "") or ""

        # basic validation
        if not note_text:
            return redirect(url_for("edit_medical_notes", pid=pid))

        # parse codes
        parsed = []
        for part in raw_codes.split(","):
            code = part.strip().upper()
            if code:
                parsed.append(code)

        if len(parsed) == 0:
            return redirect(url_for("edit_medical_notes", pid=pid, err="icd_required"))

        note_ref = patient_ref.collection("MedicalNote").document(note_id)

        # --- read current stored values to detect "no changes" ---
        old_note_text = ""
        old_codes = []

        note_doc = note_ref.get()
        if note_doc.exists:
            old_note_text = (note_doc.to_dict() or {}).get("Note", "") or ""

        # read codes from the same icd_id (or fallback to first ICD doc)
        if icd_id:
            icd_doc = note_ref.collection("ICDcode").document(icd_id).get()
            if icd_doc.exists:
                old_codes = (icd_doc.to_dict() or {}).get("Adjusted", []) or []
        else:
            icd_docs = list(note_ref.collection("ICDcode").limit(1).stream())
            if icd_docs:
                icd_id = icd_docs[0].id
                old_codes = (icd_docs[0].to_dict() or {}).get("Adjusted", []) or []

        # --- compare old vs new ---
        old_norm_text = (old_note_text or "").strip()
        new_norm_text = (note_text or "").strip()

        old_norm_codes = sorted([str(x).strip().upper() for x in (old_codes or []) if str(x).strip()])
        new_norm_codes = sorted([str(x).strip().upper() for x in (parsed or []) if str(x).strip()])

        changed = (old_norm_text != new_norm_text) or (old_norm_codes != new_norm_codes)

        # update note text (even if unchanged; harmless)
        note_ref.update({"Note": note_text})

        # update ICD doc (create if missing)
        if icd_id:
            note_ref.collection("ICDcode").document(icd_id).update({
                "Adjusted": parsed,
                "AdjustedBy": session.get("user_id"),
                "AdjustedAt": datetime.now()
            })
        else:
            new_icd_id = "icdcode_id_" + uuid.uuid4().hex[:8]
            note_ref.collection("ICDcode").document(new_icd_id).set({
                "ICD_ID": new_icd_id,
                "Adjusted": parsed,
                "Predicted": [],
                "AdjustedBy": session.get("user_id"),
                "AdjustedAt": datetime.now()
            })

        return redirect(url_for("edit_medical_notes", pid=pid, saved="1" if changed else "0"))
    

        # ---------------------------
    # DELETE ONE MEDICAL NOTE + ALL ICD CODES
    # ---------------------------
    @app.route("/delete_medical_note/<pid>/<note_id>", methods=["POST"])
    def delete_medical_note(pid, note_id):
        if 'user_id' not in session:
            return jsonify({"status": "error", "message": "Unauthorized"}), 401

        pid = (pid or "").strip()
        note_id = (note_id or "").strip()

        if not re.fullmatch(r"\d{10}", pid) or not note_id:
            return jsonify({"status": "error", "message": "Invalid patient/note id"}), 400

        patient_ref = db.collection("Patient").document(pid)
        patient_doc = patient_ref.get()
        if not patient_doc.exists:
            return jsonify({"status": "error", "message": "Patient not found"}), 404

        note_ref = patient_ref.collection("MedicalNote").document(note_id)
        note_doc = note_ref.get()
        if not note_doc.exists:
            return jsonify({"status": "error", "message": "Note not found"}), 404

        try:
            # delete all ICD docs under this note
            icd_ref = note_ref.collection("ICDcode")
            delete_collection(icd_ref, batch_size=50)

            # delete the note itself
            note_ref.delete()

            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/view_patient/<pid>", methods=["GET"])
    def view_patient(pid):
        if 'user_id' not in session:
            return redirect(url_for('Authentication.login'))

        pid = (pid or "").strip()
        if not re.fullmatch(r"\d{10}", pid):
            return redirect(url_for("dashboard"))

        patient_ref = db.collection("Patient").document(pid)
        patient_doc = patient_ref.get()

        if not patient_doc.exists:
            return redirect(url_for("dashboard"))

        pdata = patient_doc.to_dict() or {}

        form = {
            "full_name": pdata.get("FullName", ""),
            "dob": pdata.get("DOB", ""),
            "gender": pdata.get("Gender", ""),
            "phone": pdata.get("Phone", ""),
            "email": pdata.get("Email", ""),
            "address": pdata.get("Address", ""),
            "blood_type": pdata.get("BloodType", ""),
        }

        notes_out = []
        # Newest first
        notes = patient_ref.collection("MedicalNote").order_by("CreatedDate", direction="DESCENDING").stream()

        for n in notes:
            nd = n.to_dict() or {}
            note_id = n.id
            note_text = nd.get("Note", "")

            created = nd.get("CreatedDate")
            created_str = ""
            try:
                if isinstance(created, datetime):
                    created_str = created.strftime("%Y-%m-%d %H:%M")
                elif created:
                    created_str = str(created)
            except:
                created_str = ""

            icd_id = ""
            icd_codes = []
            try:
                icd_docs = list(n.reference.collection("ICDcode").limit(1).stream())
                if icd_docs:
                    icd_id = icd_docs[0].id
                    icd_data = icd_docs[0].to_dict() or {}
                    icd_codes = icd_data.get("Adjusted", []) or []
            except:
                pass

            icd_codes_str = ", ".join([str(x).strip().upper() for x in icd_codes if str(x).strip()])

            notes_out.append({
                "note_id": note_id,
                "note_text": note_text,
                "created_date": created_str,
                "icd_id": icd_id,
                "icd_codes_str": icd_codes_str
            })

        return render_template(
            "view_patient.html",
            pid=pid,
            form=form,
            notes=notes_out,
            errors=[],
        )



    # ✅ Predict
    @app.post("/predict_icd")
    def predict_icd_route():
        if "user_id" not in session:
            return jsonify({"error": "Unauthorized"}), 401

        ensure_model_loaded(app)

        data = request.get_json(silent=True) or {}
        raw_text = (data.get("note_text") or "").strip()
        if not raw_text:
            return jsonify({"status": "error", "message": "Empty note"}), 400

        # ✅ k from UI (default 20, max 50)
        try:
            k = int(data.get("top_k", 20))
        except Exception:
            k = 20
        k = max(1, min(k, 50))

        clean_text = preprocess_note_text_exact(raw_text)
        if not clean_text:
            return jsonify({"status": "error", "message": "Note became empty after preprocessing"}), 400

        enc = tokenizer(
            clean_text,
            padding="max_length",
            truncation=True,
            max_length=512,
            add_special_tokens=True,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        chunk_size = 128
        num_chunks = 512 // chunk_size

        input_ids = input_ids.view(1, num_chunks, chunk_size)
        attention_mask = attention_mask.view(1, num_chunks, chunk_size)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits)[0].detach().cpu()

        # ✅ Top-K بدل Top-5
        vals, inds = torch.topk(probs, k=k)

        predictions = []
        for s, i in zip(vals.tolist(), inds.tolist()):
            predictions.append({
                "Code": index2target.get(i, str(i)),
                "score": float(s),
            })

        return jsonify({
            "status": "success",
            "predictions": predictions,
            "default_threshold": float(BEST_THRESHOLD if BEST_THRESHOLD is not None else 0.5),
            "top_k": k
        })


    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, use_reloader=False, port=5001)


