from flask import (
    Blueprint, request, render_template, redirect,
    url_for, flash, session, current_app, jsonify
)
# Blueprint → Needed for routing
# request → Needed for reading login/signup form data
# render_template → Needed for HTML pages
# redirect, url_for → For navigation
# flash → Error messages
# session → Login sessions
# current_app → Needed for get_serializer()
# jsonify → Used in AJAX validation
from werkzeug.security import generate_password_hash, check_password_hash # Needed for hashing and verifying passwords
from firebase.Initialization import db # Needed — database reference
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature # Needed for email confirmation token system
from datetime import datetime, timezone
# used for logging timestamps or validation
import re # Needed for validating email, username formats
import os # Needed for BREVO_API_KEY, BREVO_SENDER
import requests # Needed for sending emails using Brevo API
import threading # Needed for send_email_async() background thread
import traceback



#  Blueprint
auth_bp = Blueprint("Authentication", __name__)


#  Serializer for tokens
def get_serializer():
    secret = current_app.config["SECRET_KEY"]
    return URLSafeTimedSerializer(secret)



# Load Brevo API credentials from environment variables
BREVO_API_KEY = os.environ.get("BREVO_API_KEY", "")
BREVO_SENDER_EMAIL = os.environ.get("BREVO_SENDER_EMAIL", "ouwnsystem@gmail.com")
BREVO_SENDER_NAME = os.environ.get("BREVO_SENDER_NAME", "OuwN System")
BREVO_ENDPOINT = "https://api.brevo.com/v3/smtp/email"

# sending email through brevo API
def send_brevo_email(to_email: str, subject: str, html: str, text: str = None):
    if not BREVO_API_KEY:
        print("❌ BREVO_API_KEY missing!")
        return

    # Email body sent to Brevo
    payload = {
        "sender": {"email": BREVO_SENDER_EMAIL, "name": BREVO_SENDER_NAME},
        "to": [{"email": to_email}],
        "subject": subject,
        "htmlContent": html,
    }
    if text:
        payload["textContent"] = text

    headers = {
        "api-key": BREVO_API_KEY,
        "Content-Type": "application/json",
    }

    try:
        print("Sending to Brevo...")
        res = requests.post(
            BREVO_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=30
        )
        print("Status:", res.status_code)
        print("Response:", res.text)
    except Exception as e:
        print("❌ Email send failed:", repr(e))
        traceback.print_exc()

# Sends email in a background thread so Flask does not wait for it
def send_email_async(to, subject, html, text=None):
    t = threading.Thread(target=lambda: send_brevo_email(to, subject, html, text))
    t.daemon = True
    t.start()



#  LOGIN
@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    entered_username = ""

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        entered_username = username  # Used to refill username field after errors

        try:
            # Look up user in Firestore using their UserID
            q = db.collection("HealthCareP").where("UserID", "==", username).limit(1).get()
            if not q:
                flash("Invalid username or password.", "error")
                return render_template("login.html", entered_username=username)

            user_doc = q[0]
            user = user_doc.to_dict()
           
            # Compare stored hashed password with entered password
            if not check_password_hash(user["Password"], password):
                flash("Invalid username or password.", "error")
                return render_template("login.html", entered_username=username)

            # store user session
            session["user_id"] = user_doc.id
            session["user_name"] = user["Name"]
            session["user_email"] = user["Email"]

            return redirect(url_for("dashboard"))

        except Exception:
            flash("Login failed. Try again.", "error")

    return render_template("login.html", entered_username=entered_username)


#  SIGNUP
@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    # keep entered fields to refill form if validation fails
    entered = {"first_name": "", "last_name": "", "username": "", "email": ""}

    if request.method == "POST":
        first = request.form.get("first_name", "").strip()
        last = request.form.get("last_name", "").strip()
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        entered = {"first_name": first, "last_name": last, "username": username, "email": email}

        # Email validation
        if not re.fullmatch(r"^[^@]+@[^@]+\.[A-Za-z]{2,}$", email):
            flash("Enter a valid email address.", "error")
            return render_template("signup.html", entered=entered)

        # Username format: must begin with letter, 3–32 characters
        if not re.fullmatch(r"^[A-Z][A-Za-z0-9._-]{2,31}$", username):
            flash("Username must start with a capital  letter and be 3–32 characters.", "error")
            return render_template("signup.html", entered=entered)

        # Password checks
        if (len(password) < 8 or
            not any(c.isupper() for c in password) or
            not any(c.islower() for c in password) or
            not any(c.isdigit() for c in password) or
            not any(not c.isalnum() for c in password)):
            flash("Password must include uppercase, lowercase, number, and special character.", "error")
            return render_template("signup.html", entered=entered)
        
        # Duplicate username
        all_users = db.collection("HealthCareP").select([]).stream()
        for u in all_users:
            if u.id.lower() == username.lower():
                flash("Username already exists.", "error")
                return render_template("signup.html", entered=entered)
            
         # Duplicate Email
        if db.collection("HealthCareP").where("Email", "==", email).limit(1).get():
            flash("Email already exists.", "error")
            return render_template("signup.html", entered=entered)

        # Hash password
        hashed_pw = generate_password_hash(password)

        # Create token containing full user data .NOT saved yet
        s = get_serializer()
        token = s.dumps({
            "first": first,
            "last": last,
            "username": username,
            "email": email,
            "password": hashed_pw
        }, salt="email-confirm")

        link = url_for("Authentication.confirm_email", token=token, _external=True)
        subject = "Confirm Your Email - OuwN"

        # Email content 
        html = f"""
        <html>
        <body style="font-family:'Segoe UI', Tahoma; background:#f4eefc; padding:20px;">
            <div style="max-width:600px; margin:auto; background:#fff; padding:30px; border-radius:10px;">
                <h2 style="color:#9975C1; text-align:center;">OuwN Email Confirmation</h2>
                <p>Hi {first} {last},</p>
                <p>Please confirm your email by clicking below:</p>
                <div style="text-align:center; margin:30px 0;">
                    <a href="{link}"
                        style="background:#9975C1; color:white; padding:12px 25px; border-radius:25px;">
                        Confirm Email
                    </a>
                </div>
                <p>If you didn't create an account, ignore this email.</p>
            </div>
        </body>
        </html>
        """

        send_brevo_email(email, subject, html)

        flash(" Check your email to confirm.", "success")
        return redirect(url_for("Authentication.login"))

    return render_template("signup.html", entered=entered)



#  EMAIL CONFIRMATION
@auth_bp.route("/confirm/<token>")
def confirm_email(token):
    s = get_serializer()

    # Verify token
    try:
        data = s.loads(token, salt="email-confirm", max_age=3600)
    except SignatureExpired:
        return render_template("confirm.html", msg="⚠️ Link expired. Please sign up again.")
    except BadSignature:
        return render_template("confirm.html", msg="⚠️ Invalid confirmation link.")

    username = data["username"]

     # Prevent re-confirming same account
    if db.collection("HealthCareP").document(username).get().exists:
        return render_template("confirm.html", msg="⚠️ Account already confirmed.")

    # Create account for user in firebase
    db.collection("HealthCareP").document(username).set({
        "UserID": username,
        "Name": f"{data['first']} {data['last']}",
        "Email": data['email'],
        "Password": data['password']
    })

    return render_template("confirm.html", msg="✅ Email confirmed! You can now log in.")


#  AJAX Validation
@auth_bp.route("/check")
def check_field():
    field = request.args.get("field")
    value = request.args.get("value", "").strip()
    result = {"ok": True, "valid": True, "exists": False}

    try:
         # Username validation
        if field == "username":
            result["valid"] = bool(re.fullmatch(r"^[A-Z][A-Za-z0-9._-]{2,31}$", value))
            if result["valid"]:
                value_lower = value.lower()

                # case-insensitive search
                all_users = db.collection("HealthCareP").stream()
                for u in all_users:
                    if u.id.lower() == value_lower:
                        result["exists"] = True
                        break
         # Email validation               
        elif field == "email":
            result["valid"] = bool(re.fullmatch(r"^[^@]+@[^@]+\.[A-Za-z]{2,}$", value))
            if result["valid"]:
                result["exists"] = len(db.collection("HealthCareP")
                    .where("Email", "==", value).limit(1).get()) > 0
        
         # Unknown field
        else:
            result["ok"] = False

    except Exception:
        result["ok"] = False

    return jsonify(result)
