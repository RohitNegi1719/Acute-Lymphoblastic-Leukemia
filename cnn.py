import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
train_data_dir = 'C:/Users/Rohit Negi/Desktop/Projects/Major Project/Acute Lymphoblastic Leukemia dataset/Split_Data/train'
validation_data_dir = 'C:/Users/Rohit Negi/Desktop/Projects/Major Project/Acute Lymphoblastic Leukemia dataset/Split_Data/validation'
test_data_dir = 'C:/Users/Rohit Negi/Desktop/Projects/Major Project/Acute Lymphoblastic Leukemia dataset/Split_Data/test'

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,

    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=16,  # Reduced batch size
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=16,  # Reduced batch size
    class_mode='categorical'
)

# Further Simplified Model
model = models.Sequential([
    # First convolutional block
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Second convolutional block
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Flatten and Dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')
])

# Compile the model with a reduced learning rate
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(learning_rate=0.00001),
    metrics=['accuracy']
)

# Callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=20,
    callbacks=[early_stopping, reduce_lr]
)

# Save the model
model.save('custom_cnn_simplified_model_v2.h5')

# Evaluate on test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=16,  # Consistent batch size
    class_mode='categorical'
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy:.2f}')


"""
"C:\Users\Rohit Negi\AppData\Local\Programs\Python\Python38\python.exe" "C:\Users\Rohit Negi\Desktop\Projects\Major Project\cnn.py" 
Found 2277 images belonging to 4 classes.
Found 489 images belonging to 4 classes.
2024-11-07 22:31:22.323890: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE SSE2 SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Epoch 1/20
2024-11-07 22:31:24.292534: W tensorflow/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 201867264 exceeds 10% of free system memory.
2024-11-07 22:31:24.339360: W tensorflow/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 201867264 exceeds 10% of free system memory.
2024-11-07 22:31:25.375338: W tensorflow/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 201867264 exceeds 10% of free system memory.
  1/142 [..............................] - ETA: 6:48 - loss: 2.5182 - accuracy: 0.25002024-11-07 22:31:25.685719: W tensorflow/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 201867264 exceeds 10% of free system memory.
2024-11-07 22:31:25.732151: W tensorflow/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 201867264 exceeds 10% of free system memory.
142/142 [==============================] - 191s 1s/step - loss: 1.7639 - accuracy: 0.4507 - val_loss: 1.7419 - val_accuracy: 0.1979 - lr: 1.0000e-05
Epoch 2/20
142/142 [==============================] - 193s 1s/step - loss: 1.4768 - accuracy: 0.5542 - val_loss: 2.2472 - val_accuracy: 0.2958 - lr: 1.0000e-05
Epoch 3/20
142/142 [==============================] - 199s 1s/step - loss: 1.3874 - accuracy: 0.5723 - val_loss: 1.5354 - val_accuracy: 0.4479 - lr: 1.0000e-05
Epoch 4/20
142/142 [==============================] - 194s 1s/step - loss: 1.2437 - accuracy: 0.6267 - val_loss: 0.7939 - val_accuracy: 0.8104 - lr: 1.0000e-05
Epoch 5/20
142/142 [==============================] - 195s 1s/step - loss: 1.2001 - accuracy: 0.6409 - val_loss: 0.6778 - val_accuracy: 0.8458 - lr: 1.0000e-05
Epoch 6/20
142/142 [==============================] - 195s 1s/step - loss: 1.1011 - accuracy: 0.6762 - val_loss: 0.5815 - val_accuracy: 0.8896 - lr: 1.0000e-05
Epoch 7/20
142/142 [==============================] - 189s 1s/step - loss: 1.0766 - accuracy: 0.6953 - val_loss: 0.6131 - val_accuracy: 0.8833 - lr: 1.0000e-05
Epoch 8/20
142/142 [==============================] - 191s 1s/step - loss: 1.0817 - accuracy: 0.6891 - val_loss: 0.5531 - val_accuracy: 0.9062 - lr: 1.0000e-05
Epoch 9/20
142/142 [==============================] - 190s 1s/step - loss: 1.0005 - accuracy: 0.7187 - val_loss: 0.5223 - val_accuracy: 0.9312 - lr: 1.0000e-05
Epoch 10/20
142/142 [==============================] - 191s 1s/step - loss: 0.9799 - accuracy: 0.7315 - val_loss: 0.5013 - val_accuracy: 0.9312 - lr: 1.0000e-05
Epoch 11/20
142/142 [==============================] - 190s 1s/step - loss: 0.9441 - accuracy: 0.7567 - val_loss: 0.5015 - val_accuracy: 0.9354 - lr: 1.0000e-05
Epoch 12/20
142/142 [==============================] - 190s 1s/step - loss: 0.9329 - accuracy: 0.7408 - val_loss: 0.5241 - val_accuracy: 0.9146 - lr: 1.0000e-05
Epoch 13/20
142/142 [==============================] - 190s 1s/step - loss: 0.9345 - accuracy: 0.7382 - val_loss: 0.5196 - val_accuracy: 0.9250 - lr: 1.0000e-05
Epoch 14/20
142/142 [==============================] - 190s 1s/step - loss: 0.9584 - accuracy: 0.7324 - val_loss: 0.4505 - val_accuracy: 0.9521 - lr: 1.0000e-06
Epoch 15/20
142/142 [==============================] - 190s 1s/step - loss: 0.8929 - accuracy: 0.7585 - val_loss: 0.4548 - val_accuracy: 0.9479 - lr: 1.0000e-06
Epoch 16/20
142/142 [==============================] - 190s 1s/step - loss: 0.9178 - accuracy: 0.7625 - val_loss: 0.4496 - val_accuracy: 0.9500 - lr: 1.0000e-06
Epoch 17/20
142/142 [==============================] - 191s 1s/step - loss: 0.9012 - accuracy: 0.7656 - val_loss: 0.4539 - val_accuracy: 0.9521 - lr: 1.0000e-06
Epoch 18/20
142/142 [==============================] - 193s 1s/step - loss: 0.8865 - accuracy: 0.7590 - val_loss: 0.4510 - val_accuracy: 0.9500 - lr: 1.0000e-06
Epoch 19/20
142/142 [==============================] - 193s 1s/step - loss: 0.8629 - accuracy: 0.7744 - val_loss: 0.4430 - val_accuracy: 0.9500 - lr: 1.0000e-06
Epoch 20/20
142/142 [==============================] - 191s 1s/step - loss: 0.9201 - accuracy: 0.7444 - val_loss: 0.4505 - val_accuracy: 0.9542 - lr: 1.0000e-06
C:\Users\Rohit Negi\AppData\Local\Programs\Python\Python38\lib\site-packages\keras\src\engine\training.py:3000: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(
Found 490 images belonging to 4 classes.
31/31 [==============================] - 7s 236ms/step - loss: 0.4293 - accuracy: 0.9571
Test accuracy: 0.96

Process finished with exit code 0

"""