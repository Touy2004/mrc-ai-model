# Import necessary libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, UpSampling2D, concatenate, Layer
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
import paho.mqtt.client as mqtt
import threading

# MQTT Configurations
BROKER_ADDRESS = "broker.hivemq.com"  # replace with your MQTT broker address
MQTT_TOPIC = "face-detection"  # replace with your MQTT topic

# ENCODER Block
@register_keras_serializable()
class EncoderBlock(Layer):
    def __init__(self, filters, rate, pooling=True, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.filters = filters
        self.rate = rate
        self.pooling = pooling
        self.c1 = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.drop = Dropout(rate)
        self.c2 = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.pool = MaxPool2D()

    def call(self, X):
        x = self.c1(X)
        x = self.drop(x)
        x = self.c2(x)
        if self.pooling:
            y = self.pool(x)
            return y, x
        else:
            return x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "filters": self.filters, 'rate': self.rate, 'pooling': self.pooling}

# DECODER Block
@register_keras_serializable()
class DecoderBlock(Layer):
    def __init__(self, filters, rate, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.filters = filters
        self.rate = rate
        self.up = UpSampling2D()
        self.net = EncoderBlock(filters, rate, pooling=False)

    def call(self, X):
        X, skip_X = X
        x = self.up(X)
        c_ = concatenate([x, skip_X])
        x = self.net(c_)
        return x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "filters": self.filters, 'rate': self.rate}

# ATTENTION GATE
from tensorflow.keras.layers import BatchNormalization, Add, Multiply

class AttentionGate(Layer):
    def __init__(self, filters, bn, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)
        self.filters = filters
        self.bn = bn
        self.normal = Conv2D(filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.down = Conv2D(filters, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')
        self.learn = Conv2D(1, kernel_size=1, padding='same', activation='sigmoid')
        self.resample = UpSampling2D()
        self.BN = BatchNormalization()

    def call(self, X):
        X, skip_X = X
        x = self.normal(X)
        skip = self.down(skip_X)
        x = Add()([x, skip])
        x = self.learn(x)
        x = self.resample(x)
        f = Multiply()([x, skip_X])
        if self.bn:
            return self.BN(f)
        else:
            return f

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "filters": self.filters, "bn": self.bn}

# Load custom objects and model
custom_objects = {
    'EncoderBlock': EncoderBlock,
    'DecoderBlock': DecoderBlock,
    'AttentionGate': AttentionGate
}

model = load_model('image_model.keras', custom_objects=custom_objects)
print("Model loaded successfully!")

# Global flag for processing
is_processing = False

# Function to preprocess image
def preprocess_image(image_path, target_size):
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Function to post-process predictions
def post_process(predictions, threshold=0.5):
    processed = np.where(predictions > threshold, 1, 0)
    return processed

# MQTT Callback Functions
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker successfully.")
        client.subscribe(MQTT_TOPIC)
    else:
        print("Failed to connect, return code %d\n", rc)

def on_disconnect(client, userdata, rc):
    print("Disconnected from MQTT broker.")
    try:
        client.reconnect()
    except Exception as e:
        print(f"Reconnection failed: {e}")

def on_message(client, userdata, msg):
    global is_processing
    message = msg.payload.decode()
    if message == "1" and not is_processing:
        print("Received mqtt=1, starting image processing...")
        processing_thread = threading.Thread(target=process_image)
        processing_thread.start()

# Function to process image
def process_image():
    global is_processing
    is_processing = True  # Set flag to True when processing starts
    try:
        target_size = (256, 256)  # Adjust to your model's input size
        image_path = "image.jpg"
        preprocessed_image = preprocess_image(image_path, target_size)
        predictions = model.predict(preprocessed_image)
        processed_predictions = post_process(predictions)

        # Load original image and create overlay
        original_image = load_img(image_path, target_size=target_size)
        original_image_array = img_to_array(original_image)

        blue_overlay = np.zeros_like(original_image_array)
        blue_overlay[..., 2] = 255  # Blue channel for overlay

        overlay = np.where(processed_predictions.squeeze()[:, :, np.newaxis] == 1, blue_overlay, 0)
        alpha = 0.4
        blended_image = (original_image_array * (1 - alpha) + overlay * alpha).astype(np.uint8)

        # Define a range for "blue" in RGB
        lower_blue = np.array([0, 0, 100])
        upper_blue = np.array([100, 100, 255])
        blue_mask_new = np.all((blended_image >= lower_blue) & (blended_image <= upper_blue), axis=-1)

        # Calculate blue pixel count
        blue_pixel_count = np.sum(blue_mask_new)
        total_pixel_count = blended_image.shape[0] * blended_image.shape[1]
        blue_pixel_percentage = (blue_pixel_count / total_pixel_count) * 100

        # Annotate image with text
        annotated_image = blended_image.copy()
        # cv2.putText(annotated_image, f"Total Pixels: {total_pixel_count}", (10, 20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(annotated_image, f"Flooded Pixels: {blue_pixel_count}", (10, 40),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(annotated_image, f"Flooded Area: {blue_pixel_percentage:.2f}%", (10, 60),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Save the annotated image
        output_path = 'output.png'
        cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print(f"Annotated image saved as {output_path}")
    finally:
        is_processing = False  # Reset flag after processing completes

# Initialize and start MQTT client
client = mqtt.Client()
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message
client.connect(BROKER_ADDRESS, 1883, 60)

# Run the MQTT client in a blocking loop to wait for messages and monitor connection
print("Waiting for MQTT message with mqtt=1 to start processing...")
try:
    client.loop_forever()  # This will block and keep running, disconnect will trigger on_disconnect
except KeyboardInterrupt:
    print("MQTT loop interrupted.")
finally:
    client.disconnect()