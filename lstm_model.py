import numpy as np
from tensorflow.keras.models import load_model
import paho.mqtt.client as mqtt
import json

# Load the model once when the script starts
model = load_model('lstm_model.keras')

# MQTT settings
MQTT_BROKER = "broker.hivemq.com"  # Replace with your broker address
MQTT_PORT = 1883
MQTT_TOPIC_SUBSCRIBE = "sensor/actual_data"
MQTT_TOPIC_PUBLISH = "model/prediction"

# Callback when the client connects to the broker
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker successfully.")
        print("Waiting for messages...")
    else:
        print(f"Failed to connect, return code {rc}\n")

# Callback when a message is received from the MQTT broker
def on_message(client, userdata, message):
    # Decode the message payload
    payload = message.payload.decode()
    
    # Parse JSON and extract the water level
    data = json.loads(payload)
    actual_data = float(data["water_level"])
    
    # Process the data by creating a sequence
    data_sequence = [actual_data] * 12
    data_sequence = np.array(data_sequence)

    # Create sliding windows of 3 timesteps with 16 features
    timesteps = 3
    features = 16
    X_test = []
    for i in range(len(data_sequence) - timesteps + 1):
        window = data_sequence[i:i + timesteps]
        window = np.repeat(window[:, np.newaxis], features, axis=1)
        X_test.append(window)
    X_test = np.array(X_test)

    # Make predictions with the preloaded model
    predictions = model.predict(X_test)
    result = predictions.mean()
    print(f"Prediction result: {result}")

    # Construct the message in the required format
    publish_payload = f'{{"stationID": "{data["stationID"]}", "water_level": "{data["water_level"]}", "prediction": "{float(result)}"}}'
    
    # Publish the result to another MQTT topic
    client.publish(MQTT_TOPIC_PUBLISH, publish_payload)
    
    # Print a success message after publishing
    print("Prediction successfully processed and published.")

# Set up MQTT client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Connect to the MQTT broker
client.connect(MQTT_BROKER, MQTT_PORT)
client.loop_start()

# Subscribe to the topic to receive actual_data
client.subscribe(MQTT_TOPIC_SUBSCRIBE)

try:
    # Keep the program running to listen for messages
    while True:
        pass
except KeyboardInterrupt:
    client.loop_stop()
    client.disconnect()