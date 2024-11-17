import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

# Paths
train_data_dir = 'C:/Users/Rohit Negi/Desktop/Projects/Major Project/Acute Lymphoblastic Leukemia dataset/Split_Data/train'
validation_data_dir = 'C:/Users/Rohit Negi/Desktop/Projects/Major Project/Acute Lymphoblastic Leukemia dataset/Split_Data/validation'
test_data_dir = 'C:/Users/Rohit Negi/Desktop/Projects/Major Project/Acute Lymphoblastic Leukemia dataset/Split_Data/test'

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

# Generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),  # Increasing input size for transfer learning
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Transfer Learning with VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
"""
weights='imagenet': Loads the pre-trained weights of VGG16, which were trained on the ImageNet dataset.
include_top=False: Excludes the top fully connected layers of VGG16, as we'll add our own custom classification layers.
input_shape=(224, 224, 3): Specifies the input shape for the model, which is a 224x224 pixel image with 3 color channels (RGB).
"""
base_model.trainable = False  # Freeze the base model layers
#during training, the weights of the VGG16 layers will not be updated
# as the pre-trained weights are already well-suited for feature extraction

# Build the model
model = models.Sequential([
    base_model   # pre-trained VGG16 model
    layers.Flatten(), #converts the 3D o/p of the convolutional base((batch_size, height, width, channels)) into a 1D vector
    layers.Dense(256, activation='relu'), #First Dense Layer: Has 256 neurons with ReLU activation.
    layers.Dropout(0.3), # Dropout Layer:Randomly drops out 30% of neurons during training.
    layers.BatchNormalization(), #Batch Normalization Layer: Normalizes the input to each layer.
    layers.Dense(128, activation='relu'), #Second Dense Layer: Has 128 neurons with ReLU activation.
    layers.Dropout(0.3), #Second Dropout layer  with 30% dropout rate
    layers.Dense(4, activation='softmax')  # 4 classes (benign, early , pre, pro)
])

# A[VGG16]->B{Flatten}->C[Dense 256]->D{Dropout 0.3}->E[Batch Normalization]->F[Dense 128]->G{Dropout 0.3}->H[Output(4)]

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
              metrics=['accuracy'])

# Callbacks
#for  customising the behavior of the training process
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)


# Save the model
model.save('optimized_cnn_model.h5')

# Evaluate on test data
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy:.2f}')
"""16/16 [==============================] - 46s 3s/step - loss: 0.0821 - accuracy: 0.9816
Test accuracy: 0.98"""