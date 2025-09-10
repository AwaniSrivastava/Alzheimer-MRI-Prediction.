import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(224,224,3), num_classes=4):
    """
    Build CNN model using Transfer Learning (ResNet50) with Functional API.
    Compatible with TensorBoard Graph visualization.
    """
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)

    # Base ResNet50 (pretrained on ImageNet)
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs
    )
    base_model.trainable = False  # freeze base layers initially

    # Custom classification head
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Build model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile
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
        model: keras Functional model
        fine_tune_at: number of layers from the end to unfreeze
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name.startswith("resnet50"):
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Base ResNet50 not found in model.")

    # Unfreeze the last N layers
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    # Recompile with lower LR for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
