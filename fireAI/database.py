from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
uri = "mongodb+srv://sydneywilby:csTLuCfjkE3UMd1y@firebotdetection.oiwyq.mongodb.net/?retryWrites=true&w=majority&appName=FireBotDetection"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
db = client["fire_detection"]
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
    print(db["cameras"].find())

    # for loop to print all the cameras

    for camera in db["cameras"].find():
        print(camera["camera_id"])

except Exception as e:
    print(e)

