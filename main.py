import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('model.h5')
model.summary()


st.cache_data()
def load_model():
  model=tf.keras.models.load_model('model.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Tuberculosis classification
         """
         )

file = st.file_uploader("Please upload a chest ct scan image file", type=["jpg", "png"])

st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
        # img = cv2.imread(image_data)
        img = cv2.resize(image_data,[96,96])
        img = img[np.newaxis,...]
        prediction= model.predict(img, verbose=1)
        
        # cm_plot_labels = ['Normal', 'Tuberculosis']
        # y_pred = predictions.argmax(axis=1)
        # print(cm_plot_labels[y_pred[0]])        
        return prediction

if file is None:
    st.text("Please upload an image file")
else:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    # img = cv2.imread(file)
    st.image(img, use_column_width=True)
    cm_plot_labels = ['Normal', 'Tuberculosis']
    prediction = import_and_predict(img, model)
    score = tf.nn.softmax(prediction[0])
    
    
    # st.write(prediction)
    # st.write(score)
    st.write(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(cm_plot_labels[np.argmax(score)], 100 * np.max(score))
)
