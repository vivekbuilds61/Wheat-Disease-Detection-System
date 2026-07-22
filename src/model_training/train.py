import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

RAW_DATA_DIR = "data/raw"
MODEL_SAVE_PATH = "models/classification/mobilenet_v2_wheat.h5"
LABELS_PATH = "models/classification/labels.txt"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

def main():
    if not os.path.exists(RAW_DATA_DIR):
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_DIR}")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=15,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_gen = train_datagen.flow_from_directory(
        RAW_DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training"
    )

    val_gen = train_datagen.flow_from_directory(
        RAW_DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation"
    )

    class_names = list(train_gen.class_indices.keys())
    print("Classes:", class_names)

    os.makedirs(os.path.dirname(LABELS_PATH), exist_ok=True)
    with open(LABELS_PATH, "w") as f:
        for c in class_names:
            f.write(c + "\n")

    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    out = Dense(len(class_names), activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=out)

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    model.save(MODEL_SAVE_PATH)
    print(f"Model saved: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
