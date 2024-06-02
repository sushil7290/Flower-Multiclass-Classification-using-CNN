import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.regularizers import l2

# Define the main directory
main_dir = r"path to your dataset main directory"

# List directories
valid_dir = os.path.join(main_dir, 'Valid')
train_dir = os.path.join(main_dir, 'train')

batch_size = 32
target_size= ( 224, 224 )


# ----------------------
# Get the class names
class_names = os.listdir(train_dir)

# Adjust the figure size and layout to accommodate the images
plt.figure(figsize=(10, 10))

# Load and display sample images from the training directory
for class_name in class_names:
    class_dir = os.path.join(train_dir, class_name)
    for i, filename in enumerate(os.listdir(class_dir)[:1]):  # Display only first 1 images from each class
        img_path = os.path.join(class_dir, filename)
        img = image.load_img(img_path, target_size=(128, 128))
        ax = plt.subplot(len(class_names), 3, len(class_names) * i + class_names.index(class_name) + 1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis("off")

# Show the plot
plt.show()


# Creating a data generator for training set with grayscale images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_datagen_flow = train_datagen.flow_from_directory(
    train_dir,
    batch_size=batch_size,
    target_size=target_size,  # Set target size for grayscale images
    class_mode='sparse',
)

val_datagen_flow = train_datagen.flow_from_directory(
    valid_dir,
    batch_size=batch_size,
    target_size=target_size,  # Set target size for grayscale images
    class_mode='sparse',

)



# Check if the generators are correctly set up
print(f'Found {train_datagen_flow.samples} images in training directory.')
print(f'Found {val_datagen_flow.samples} images in validation directory.')

# Ensure that the generators are not empty before proceeding with model training
if train_datagen_flow.samples == 0 or val_datagen_flow.samples == 0:
    raise ValueError("One of the data generators is empty. Check your data directory paths and ensure there are images present.")


# Define the model
model = Sequential([
    Conv2D(128, (3, 3), activation='relu', input_shape=(224, 224, 3),kernel_regularizer=l2(0.01)),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu',kernel_regularizer=l2(0.01)),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu',kernel_regularizer=l2(0.01)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu',kernel_regularizer=l2(0.01)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu',kernel_regularizer=l2(0.01)),
    MaxPooling2D(2, 2),
    BatchNormalization(momentum=0.9),
    Conv2D(16, (3, 3), activation='relu',kernel_regularizer=l2(0.01)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu',kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(128, activation='relu',kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(17, activation='softmax')
])



model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
trained_model = model.fit(train_datagen_flow, epochs=20, batch_size=batch_size, validation_data=val_datagen_flow, callbacks=[early_stopping], verbose=1)

# Save the model
model_dir = r"path to save your model with '.h5' extension"
model.save(model_dir)

# Evaluate the model on the validation set
validation_results = model.evaluate(val_datagen_flow)

# Print performance metrics
print("Validation Loss:", validation_results[0])
print("Validation Accuracy:", validation_results[1])

# Predict classes for the validation set
true_classes = val_datagen_flow.classes
validation_predictions = model.predict(val_datagen_flow)
predicted_classes = np.argmax(validation_predictions, axis=1)

# Generate classification report
class_labels = list(val_datagen_flow.class_indices.keys())
print("Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Generate confusion matrix
print("Confusion Matrix:")
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print(conf_matrix)


# Plot ROC-AUC curve and calculate EER
fpr = dict()
tpr = dict()
roc_auc = dict()
eer = dict()
for i in range(1):
    fpr[i], tpr[i], _ = roc_curve(true_classes, validation_predictions[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute EER
    fnr = 1 - tpr[i]
    eer[i] = fpr[i][np.nanargmin(np.absolute((fnr - fpr[i])))]
    
    plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve (class {i}) (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Calculate the average EER
average_eer = np.mean(list(eer.values()))
print("Equal Error Rate (EER):", average_eer)

# Calculate Correct Rejection Rate (CRR)
crr = 1 - (average_eer / 2)
print("Correct Rejection Rate (CRR):", crr)


# Plot training and validation accuracy
plt.plot(trained_model.history['accuracy'], label='Training Accuracy')
plt.plot(trained_model.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot training and validation loss
plt.plot(trained_model.history['loss'], label='Training Loss')
plt.plot(trained_model.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels, rotation=45)
plt.yticks(tick_marks, class_labels)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

