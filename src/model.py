import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(224,224,3), num_classes=4, fine_tune_at=-30):
    """
    Build a CNN model using Transfer Learning (ResNet50) + Fine-tuning.

    Args:
        input_shape: shape of input images
        num_classes: number of output classes
        fine_tune_at: number of layers from the end to unfreeze for fine-tuning
    """

    # Load ResNet50 base model (without top layer)
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Freeze all layers initially
    base_model.trainable = False

    # Build top classifier
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    # Compile initial model (train only top classifier first)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def fine_tune_model(model, fine_tune_at=-30):
    """
    Unfreeze part of ResNet50 for fine-tuning.

    Args:
        model: compiled keras model
        fine_tune_at: number of layers from the end to unfreeze
    """

    base_model = model.layers[0]  # ResNet50 backbone

    # Unfreeze the last N layers
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    # Recompile with smaller learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
