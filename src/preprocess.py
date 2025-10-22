import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def load_data(base_dir, img_size=(224, 224), batch_size=32):
    """
    Load and preprocess MRI dataset with train/test folders.
    base_dir should contain:
        train/
            MildDemented/
            ModerateDemented/
            VeryMildDemented/
            NonDemented/
        test/
            MildDemented/
            ModerateDemented/
            VeryMildDemented/
            NonDemented/
    """

    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,  # 80% training, 20% validation
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # Only rescaling for test set
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Training generator
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    # Validation generator
    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    # Test generator
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False  # important for evaluation
    )

    return train_gen, val_gen, test_gen