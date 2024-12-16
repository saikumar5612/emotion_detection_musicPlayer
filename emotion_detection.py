import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Build the CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(7, activation='softmax')  # 7 classes for emotions
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load dataset and preprocess
def load_dataset():
    train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        'path_to_train_data',
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical'
    )
    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        'path_to_validation_data',
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical'
    )
    return train_gen, val_gen

# Train and save the model
def train_model():
    train_gen, val_gen = load_dataset()
    model = build_model()
    model.fit(train_gen, validation_data=val_gen, epochs=25)
    model.save('emotion_cnn_model.h5')

if __name__ == '__main__':
    train_model()
