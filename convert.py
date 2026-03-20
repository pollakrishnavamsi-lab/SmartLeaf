import tensorflow as tf

# Load your saved .keras model
model = tf.keras.models.load_model("plant_disease_model.keras")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: optimization (recommended for mobile)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save the .tflite file
with open("leaf_model.tflite", "wb") as f:
    f.write(tflite_model)

print(" Conversion Successful! leaf_model.tflite created.")