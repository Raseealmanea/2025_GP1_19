from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app, session
# Blueprint → Needed
# render_template → Needed for HTML pages
# request → Needed for reading the email + new password form
# flash → Error messages
# redirect, url_for → Required for navigation after reset
# current_app → Needed for token serializer
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature # Needed for generating + verifying password reset tokens
from werkzeug.security import generate_password_hash # Needed for hashing the new password
from firebase.Initialization import db # Needed for finding/updating users
import os # Needed for BREVO API configuration
import re # Needed for validating email format + password rules
import threading # Needed for async email sending (send_email_async)
import traceback # Helpful for debugging errors — optional but useful
import requests# Needed to call Brevo API for sending the reset email


# Blueprint for all password reset related routes
reset_bp = Blueprint("auth_reset", __name__, url_prefix="/auth/reset")


# Token Serializer
def get_serializer():
    secret = current_app.config.get("SECRET_KEY")
    return URLSafeTimedSerializer(secret)


# Load Brevo email credentials from environment variables
BREVO_API_KEY = os.environ.get("BREVO_API_KEY", "")
BREVO_SENDER_EMAIL = os.environ.get("BREVO_SENDER_EMAIL", "ouwnsystem@gmail.com")
BREVO_SENDER_NAME = os.environ.get("BREVO_SENDER_NAME", "OuwN System")
BREVO_ENDPOINT = "https://api.brevo.com/v3/smtp/email"

def build_password_action_email(reset_link: str, username: str, mode: str = "reset"):
    """Build password reset/change email content using the same layout."""
    is_change = mode == "change"
    noun_phrase = "Password Change" if is_change else "Password Reset"
    action_phrase = "change your password" if is_change else "reset your password"
    requested_phrase = "change your password" if is_change else "reset your password"
    button_label = "Change Password" if is_change else "Reset Password"
    subject = f"OuwN • {noun_phrase} Link"
    text_body = f"{button_label}: {reset_link}"

    html_body = f"""
        <html>
        <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    color: #2d004d; background: #f4eefc; padding: 20px;">
            <div style="max-width: 600px; margin: auto; background: #fff; border-radius: 10px;
                        padding: 30px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">

            <h2 style="color: #9975C1; text-align: center;">OuwN {noun_phrase}</h2>

            <p>Hi {username},</p>

            <p>You requested to {requested_phrase}. Click the button below to {action_phrase}:</p>

            <div style="text-align: center; margin: 30px 0;">
                <a href="{reset_link}"
                style="background: #9975C1; color: white; padding: 12px 25px;
                        text-decoration: none; border-radius: 25px; font-weight: bold;">
                {button_label}
                </a>
            </div>

            <p>If you didn't request this, you can ignore this email.</p>

            <p>Thanks,<br><strong>OuwN Team</strong></p>
            </div>
        </body>
        </html>
        """

    return subject, text_body, html_body

# sending email through Brevo API
def send_brevo_email(to_email: str, subject: str, html: str, text: str = None):
    """Send email using Brevo API."""
    if not BREVO_API_KEY:
        print("❌ Missing BREVO_API_KEY")
        return False

    # API payload with the message content
    payload = {
        "sender": {"email": BREVO_SENDER_EMAIL, "name": BREVO_SENDER_NAME},
        "to": [{"email": to_email}],
        "subject": subject,
        "htmlContent": html
    }

    if text:
        payload["textContent"] = text

    headers = {
        "api-key": BREVO_API_KEY,
        "Content-Type": "application/json"
    }

    try:
        print(f"📨 Sending reset email → {to_email}")
        res = requests.post(BREVO_ENDPOINT, json=payload, headers=headers)

        # Log errors if any
        if res.status_code >= 400:
            print("❌ BREVO ERROR:", res.text)
            return False
        else:
            print("✅ Email sent:", res.json())
            return True

    except Exception as e:
        print("❌ Brevo exception:", e)
        traceback.print_exc()
        return False


# Runs the email-sending function in a background thread
def send_email_async(to, subject, html, text=None):
    thread = threading.Thread(target=lambda: send_brevo_email(to, subject, html, text))
    thread.daemon = True
    thread.start()


# Password Reset Request Page
@reset_bp.route("/request", methods=["GET", "POST"])
def reset_request():
    message = ""

    if request.method == "POST":
        email = request.form.get("email", "").strip()

        # Email format validation
        if not re.fullmatch(r"^[^@]+@[^@]+\.[A-Za-z]{2,}$", email):
            return render_template("reset_password.html", message="Please enter a valid email address.")

        # Check if email exists
        users = db.collection("HealthCareP").where("Email", "==", email).get()
        if not users:
            return render_template("reset_password.html", message="✅ If an account exists with this email, a password reset link has been sent. Please check your inbox.")

        user_doc = users[0].to_dict()
        
        #  Create password reset token
        s = get_serializer()
        token = s.dumps({"email": email, "mode": "reset"}, salt="password-reset")
        reset_link = url_for("auth_reset.reset_password", token=token, _external=True)

        username = user_doc.get("Name", "User")
        subject, text_body, html_body = build_password_action_email(reset_link, username, mode="reset")

        # Send email 
        try:
            if send_brevo_email(email, subject, html_body, text_body):
                message = "✅ If an account exists with this email, a password reset link has been sent. Please check your inbox."
            else:
                message = "Failed to send email. Please try again later."
        except Exception as e:
            print("❌ Reset email error:", e)
            message = "Failed to send email. Please try again later."

    return render_template("reset_password.html", message=message)


# Reset Password Form
@reset_bp.route("/<token>", methods=["GET", "POST"])
def reset_password(token):
    s = get_serializer()

    # Verify token
    try:
        data = s.loads(token, salt="password-reset", max_age=3600)
        email = data.get("email")
        mode = data.get("mode", "reset")
    except SignatureExpired:
        return render_template("reset_password.html", message="⚠️ The reset link has expired.")
    except BadSignature:
        return render_template("reset_password.html", message="⚠️ Invalid reset link.")

    # Handle password update
    if request.method == "POST":
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        # Match check
        if password != confirm:
            return render_template("reset_token.html", message="Passwords do not match.")

        # Strength check
        rules = {
            "length": len(password) >= 8,
            "upper": bool(re.search(r"[A-Z]", password)),
            "lower": bool(re.search(r"[a-z]", password)),
            "digit": bool(re.search(r"\d", password)),
            "special": bool(re.search(r"[^A-Za-z0-9]", password)),
        }

        if not all(rules.values()):
            return render_template(
                "reset_token.html",
                message="Password must be 8+ characters and include upper, lower, digit, and special symbol."
            )

        # Find the user by email
        user_docs = db.collection("HealthCareP").where("Email", "==", email).get()

        if not user_docs:
            return render_template("reset_token.html", message="User not found.")

        # Update password
        user_ref = user_docs[0].reference
        user_ref.update({"Password": generate_password_hash(password)})

        if mode == "change" and session.get("user_id"):
            flash("✅ Your password has been changed successfully.", "success")
            return redirect(url_for("dashboard"))

        flash("✅ Your password has been reset! Please log in.", "success")
        return redirect(url_for("Authentication.login"))

    return render_template("reset_token.html", message="", mode=mode)
