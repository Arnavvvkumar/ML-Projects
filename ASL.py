import os
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


img_size   = (64, 64)
batch_size = 32
train_dir  = "/Users/arnavkumar/Documents/ASL dataset/asl_alphabet_train/asl_alphabet_train"
model_path = "models/asl_classifier.h5"

os.makedirs(os.path.dirname(model_path), exist_ok=True)


datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.1 
)


letters_24 = [chr(c) for c in range(65, 91) if chr(c) not in ("J", "Z")]

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    classes=letters_24,
    subset='training'     
)

val_gen = datagen.flow_from_directory(
    train_dir,            
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    classes=letters_24,
    subset='validation'  
)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*img_size, 3)),
    BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    Dropout(0.3),
    layers.Dense(128, activation='relu'),
    Dropout(0.3),
    layers.Dense(len(train_gen.class_indices), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    verbose=1
)

model.save(model_path)
print(f"✅ Trained model saved to {model_path}")
