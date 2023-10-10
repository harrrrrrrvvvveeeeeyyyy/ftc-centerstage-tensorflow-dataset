import numpy as np
import os

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf

import pathlib.Path as _path

assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

spec = model_spec.get('efficientdet_lite0')

'''

Model architecture   Size(MB)	Latency	Average Precision***
EfficientDet-Lite0	4.4	37	25.69%
EfficientDet-Lite1	5.8	49	30.55%
EfficientDet-Lite2	7.2	69	33.97%
EfficientDet-Lite3	11.4	116	37.70%
EfficientDet-Lite4	19.9	260	41.96%

'''

csv_dir = _path("data/csv/data.csv").absolute()

train_data, validation_data, text_data = object_detector.DataLoader.from_csv(csv_dir)

model = object_detector.create(train_data, 

model_spec=spec, 

batch_size=1, 

tain_whole_model=True, 

validation_data=validation_data)

model.evaluate(test_data)

print("Training Finished, exporting the model")

model.export(export_dir='export')

#model.evaluate_tflite('model.tflite', test_data)