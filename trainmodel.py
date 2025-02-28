import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 📂 Dataset Path (Change if needed)
dataset_path = r"C:\Users\varsh\Desktop\project slp\Dataset"

# 🔄 Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# 🏋️ Load Training Data
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),  # ✅ Changed to 224x224
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

# 🔬 Load Validation Data
val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),  # ✅ Changed to 224x224
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# 📊 Number of Classes
num_classes = len(train_generator.class_indices)
print("Classes:", train_generator.class_indices)

# 🏗️ **Use Transfer Learning - MobileNetV2**
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # Freeze base model layers

# 🔥 New Layers on Top
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# 🛠️ Compile the Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

# 🏅 Save Best Model
checkpoint_path = "best_model.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')

# 📉 Learning Rate Reduction
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# ⏳ Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 🏋️ Train Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[checkpoint, lr_reducer, early_stopping]
)

# 📊 Evaluate Model
loss, acc = model.evaluate(val_generator)
print(f"Final Validation Accuracy: {acc:.2f}")

# ✅ Save Model
model.save("best_model.h5")
print(" Model saved successfully!")




















