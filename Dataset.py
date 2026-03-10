import os
import tensorflow as tf
import numpy as np

# https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=pt-br#writing_a_tfrecord_file
def _bytes_feature(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()
  elif isinstance(value, str):
    value = value.encode("utf-8")
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def load_data(dataset_path):
    subsets = []
    for subset in ["train", "test"]:
        subset_path = os.path.join(dataset_path, subset)
        samples = []
        for track in os.listdir(subset_path):
            track_path = os.path.join(subset_path, track)
            if not os.path.isdir(track_path):
                continue
            paths = {
                0: os.path.join(track_path, "mixture.wav"),
                1: os.path.join(track_path, "vocals.wav"),
                2: os.path.join(track_path, "bass.wav"),
                3: os.path.join(track_path, "drums.wav"),
                4: os.path.join(track_path, "other.wav")
            }
            samples.append(paths)
        subsets.append(samples)
    return subsets

def parse_data(dsd,writer):
    for data in dsd:
        for key in data.keys():
            # waveform, sample_rate = torchaudio.load(data[key])
            # waveform = waveform.numpy().flatten()
            features = {
                "type": _int64_feature(key),
                "path": _bytes_feature(data[key]),
            }
            row = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(row.SerializeToString())

def Get_dataset(dataset_path):
    PATH_DFRECORDS = "./TFRecords"
    tf_train, tf_test = load_data(dataset_path)
    tf_dataset = {
        "test": tf_test,
        "train": tf_train
    }
    for name, content in tf_dataset.items():
        with tf.io.TFRecordWriter(path=f"{PATH_DFRECORDS}/{name}.tfrecords") as writer:
            row = parse_data(content,writer)
    
Get_dataset("./audio")

"""
Boas práticas:
- Salvar o local dos arquivos, ao invés de salvar o arquivo propriamente
- Salvar cada 'linha' processada por vez, para não sobrecarregar na hora de criar o arquivo TFRecords
"""