import numpy as np
import tensorflow as tf
import cv2
import pathlib

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="export/model.tflite")

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()

ref: dict = {0:"daisy", 1:"dandelion", 2:"roses", 3:"sunflowers", 4:"tulips"}

for file in pathlib.Path("testing_img").iterdir():
    
    # read and resize the image
    img = cv2.imread(r"{}".format(file.resolve()))
    new_img = cv2.resize(img, (224, 224))
    
    # input_details[0]['index'] = the index which accepts the input
    interpreter.set_tensor(input_details[0]['index'], [new_img])
    
    # run the inference
    interpreter.invoke() 
    
    output_data = list(interpreter.get_tensor(output_details[0]['index'])[0])

    name:str = ref[output_data.index(max(output_data))]

    if max(output_data) >= 200:

       print(name)

    else:

       print("Not recognized")
