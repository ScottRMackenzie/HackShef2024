import json
from os import environ as env
from urllib.parse import quote_plus, urlencode
from authlib.integrations.flask_client import OAuth
from dotenv import find_dotenv, load_dotenv
from flask import Flask, redirect, render_template, session, url_for, jsonify,request,send_from_directory, abort
import os
import base64
from pymongo import MongoClient
from datetime import datetime, timedelta
import cv2
import numpy as np

from keras.models import load_model

print("Starting up...")

# Load the trained model
model = load_model('fire_detection_model.keras')

# Load environment variables from .env file
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = env.get("SECRET_KEY")

# MongoDB setup
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
uri = "mongodb+srv://sydneywilby:csTLuCfjkE3UMd1y@firebotdetection.oiwyq.mongodb.net/?retryWrites=true&w=majority&appName=FireBotDetection"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


db = client["fire_detection"]
cameras_collection = db["cameras"]
alerts_collection = db["alerts"]

# Initialize OAuth client for Auth0
oauth = OAuth(app)

# Register Auth0 with OAuth
oauth.register(
    "auth0",
    client_id=env.get("AUTH0_CLIENT_ID"),
    client_secret=env.get("AUTH0_CLIENT_SECRET"),
    client_kwargs={
        "scope": "openid profile email",  # Request openid, profile, and email
    },
    server_metadata_url=f'https://{env.get("AUTH0_DOMAIN")}/.well-known/openid-configuration',  # This provides jwks_uri
)


# Controllers / Routes
@app.route("/")
def home():
    return render_template("home.html")
    # Display home page with user info if logged in
    # if session.get("user") is None: 
    #     render_template("index.html")
    # else:
    #     return render_template(
    #         "index.html",
    #         session=session.get("user"),
    #         pretty=json.dumps(session.get("user"), indent=4),
    #     )


@app.route("/callback", methods=["GET", "POST"])
def callback():
    # Auth0 callback to authorize access token
    token = oauth.auth0.authorize_access_token()
    session["user"] = token
    return redirect("/")  # Redirect to home page after successful login


@app.route("/login")
def login():
    # Redirect user to Auth0 login page
    return oauth.auth0.authorize_redirect(
        redirect_uri=url_for("callback", _external=True)  # Redirect to callback route after login
    )


@app.route("/logout")
def logout():
    # Step 1: Clear the local session
    session.clear()

    # Step 2: Redirect to the Auth0 logout URL
    auth0_domain = os.getenv("AUTH0_DOMAIN")
    client_id = os.getenv("AUTH0_CLIENT_ID")
    return_to = url_for("home", _external=True)  # Redirect back to your homepage after logout

    # Construct the Auth0 logout URL
    auth0_logout_url = (
        f"https://{auth0_domain}/v2/logout?"
        + urlencode(
            {
                "returnTo": return_to,
                "client_id": client_id,
            },
            quote_via=quote_plus,
        )
    )

    return redirect(auth0_logout_url)


# Secured route example
@app.route("/dashboard")
def profile():
    # Check if user is logged in (i.e., session contains user info)
    if "user" not in session:
        return redirect(url_for("login"))  # If not logged in, redirect to login

    cameras = list(cameras_collection.find())
    alerts = list(alerts_collection.find())
    return render_template("dashboard.html", user=session["user"],cameras = cameras,alerts = alerts)  # Display profile page


# @app.route('/camera/update/<camera_id>', methods=['POST'])
# def update_camera_access(camera_id):
#     now = datetime.now()
#     cameras_collection.update_one(
#         {"camera_id": camera_id},
#         {"$set": {"last_accessed": now, "image": ""}},
#         upsert=True
#     )
#     return {"status": "success"}, 200


@app.route("/camera/live/<int:camera_id>", methods=['GET'])
def view_camera(camera_id):
    # Check if user is logged in (i.e., session contains user info)
    if "user" not in session:
        return redirect(url_for("login"))  # If not logged in, redirect to login
    
    if camera_id is None:
        return "Camera ID is required", 400
    
    return render_template("camera.html", user=session["user"],camera_id = camera_id)  # Display profile page

# Directory to save received images (optional)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/camera/getview/<camera_id>', methods=['GET'])
def get_camera_image(camera_id):
    if "user" not in session:
        return redirect(url_for("login"))
    
    camera = cameras_collection.find_one({"camera_id": int(camera_id)})
    
    if camera:
        return camera["image"], 200
    else:
        abort(404, description="Image not found")

@app.route('/camera/view/<camera_id>', methods=['GET'])
def view_camera_image(camera_id):
    if "user" not in session:
        return redirect(url_for("login"))  # If not logged in, redirect to login
    return render_template("camerastream.html", user=session["user"],camera_id = camera_id)  # Display profile page

@app.route('/upload/camera/<int:camera_id>', methods=['POST'])
def upload(camera_id):
    if "user" not in session:
        return redirect(url_for("login"))

    if camera_id is None:
        return "Camera ID is required", 400

    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Process base64 image
        image_data = data['image']
        image_bytes = base64.b64decode(image_data.split(',')[1])
        
        # Convert bytes to numpy array for prediction
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run prediction
        img_resized = cv2.resize(img, (224, 224)) / 255.0
        pred = model.predict(np.expand_dims(img_resized, axis=0))
        pred_prob = float(pred[0][0])

        # Save image and update database
        cameras_collection.update_one(
            {"camera_id": camera_id},
            {
                "$set": {
                    "image": image_bytes,
                    "last_updated": datetime.now(),
                    "fire_probability": pred_prob
                }
            },
            upsert=True
        )

        # Create alert if fire probability is high
        if pred_prob > 0.9:
            alerts_collection.insert_one({
                "camera_id": camera_id,
                "timestamp": datetime.now(),
                "fire_prob": pred_prob
            })

        return jsonify({
            'message': 'Image processed successfully',
            'fire_probability': pred_prob
        }), 200

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/camera/delete/<camera_id>', methods=['DELETE'])
def delete_camera(camera_id):
    if "user" not in session:
        return redirect(url_for("login"))  # If not logged in, redirect to login
    
    # Check if the camera_id exists in the MongoDB database
    camera = cameras_collection.find_one({"camera_id": int(camera_id)})
    
    if camera:
        # Camera found, delete it
        cameras_collection.delete_one({"camera_id": camera_id})
        return {"message": f"Camera with ID {camera_id} has been deleted."}, 200
    else:
        # Return a 404 error if the camera does not exist in the database
        abort(404, description="Camera not found in database")

@app.route('/alerts')
def alerts():
    if "user" not in session:
        return redirect(url_for("login"))  # If not logged in, redirect to login
    # Simulate fire detection alert

    # Get all alerts from the MongoDB database that are newer than 20 seconds
    now = datetime.now()
    # Get all alerts from the MongoDB database that are less than than 2 minutes old
    five_minutes_ago = now - timedelta(minutes=5)
    alerts = list(alerts_collection.find({"timestamp": {"$gt": five_minutes_ago}}))

    # Convert ObjectId to string
    for alert in alerts:
        alert["_id"] = str(alert["_id"])

    return jsonify(alerts)


if __name__ == "__main__":
    # Run Flask app on the specified port
    app.run(host="0.0.0.0")