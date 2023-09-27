import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Configurações
data_dir = './Program/data' # Pasta na qual vão estar as folders com as respectivas fodos da dataset das plantas
img_height, img_width = 600, 400 # Resolução das imagens ( Deve-se manter um padrão para todo dataset )
batch_size = 32
num_classes = 4 # Número de classes, no nosso caso 20 já que são 10 tipos de plantas, cada uma com estado bom ou doente, dando assim 20 classes diferentes.
epochs = 5 # Número de epocas de treinamento da ia

train_datagen = ImageDataGenerator(
    rescale= 1.0/255,
    validation_split=0.2, # 80% para treinamento e 20% para teste
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Carregar e preparar o conjunto de dados
generate_train = train_datagen.flow_from_directory(
    data_dir,
    target_size=(int(img_height), int(img_width)),
    batch_size=batch_size,
    class_mode='categorical',  # Para classificação multi-classe
    subset='training'  # Este é o subconjunto de treinamento
)

generate_test = train_datagen.flow_from_directory(
    data_dir,
    target_size=(int(img_height), int(img_width)),
    batch_size=batch_size,
    class_mode='categorical',  #  Para classificação multi-classe
    subset='validation'
)  # Este é o subconjunto de validação

model = Sequential ([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='accuracy')

history = model.fit(
    generate_train,
    epochs=epochs,
    validation_data=generate_test
    )

model.save('leaf_model.h1')


# Obtenha os valores de precisão e perda do histórico
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Crie um gráfico de precisão
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), train_accuracy, label='Precisão de Treinamento')
plt.plot(range(epochs), val_accuracy, label='Precisão de Validação')
plt.title('Precisão de Treinamento e Validação')
plt.xlabel('Épocas')
plt.ylabel('Precisão')
plt.legend()

# Crie um gráfico de perda
plt.subplot(1, 2, 2)
plt.plot(range(epochs), train_loss, label='Perda de Treinamento')
plt.plot(range(epochs), val_loss, label='Perda de Validação')
plt.title('Perda de Treinamento e Validação')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

# Mostra os gráficos
plt.tight_layout()
plt.show()

