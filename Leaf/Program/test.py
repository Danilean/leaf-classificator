import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import load_model

def predict_image(model, image_path, img_height, img_width):
  img = load_img(image_path, target_size=(img_height, img_width))
  img_array = img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0) / 255
  prediction = model.predict(img_array)
  return prediction

def get_class_label (prediction, class_indices):
  class_label = None
  max_prob = np.max(prediction)
  for label, index in class_indices.items():
    if prediction[0][index] == max_prob:
      class_label = label
      break
  return class_label

model = load_model('leaf_model.h1')

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
      './Program/data',
      target_size=(341, 225),
      batch_size=32,
      class_mode = 'categorical'
)

class_indices = train_generator.class_indices

image_path = './Program/teste/MangoRuim02.JPG'

prediction = predict_image(model, image_path, 341, 225)
class_label = get_class_label(prediction, class_indices)

print(f"A imagem '{image_path}' foi classificada como '{class_label}' com probabilidade '{np.max(prediction) * 100:.2f}%.")