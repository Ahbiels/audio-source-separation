import os
import tensorflow as tf
import numpy as np
from Utils import *

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

num_shards = 3

def load_data_path(dataset_path):
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

def save_data(type, paths, writer):
    data_waveform, rate_of_sample = audio_to_waveform(paths)
    data_waveform = downmix_to_mono(data_waveform)
    data_waveform = trim_audio(data_waveform)
    data_waveform, rate_of_sample = resample(data_waveform, rate_of_sample)
   
    data_waveform = data_waveform.to(torch.float32)
    data_waveform = tf.io.serialize_tensor(data_waveform)
    features = {
        "type": _int64_feature(type),
        "waveform": _bytes_feature(data_waveform.numpy()),
    }
    row = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(row.SerializeToString())

def Get_dataset(dataset_path):
    tf_train, tf_test = load_data_path(dataset_path)
    
    df = {
        "train": tf_train,
        "test": tf_test
    }

    for type, tf_type in df.items():
        PATH_DFRECORDS = f"./TFRecords/{type}"
        if not os.path.exists(PATH_DFRECORDS):
            os.makedirs(PATH_DFRECORDS)
        quant_paths = len(tf_type)
        im_per_shard = 3
        num_shards = int(quant_paths/im_per_shard) + 1
        
        
        for shard in range(num_shards):
            output_filename = os.path.join(PATH_DFRECORDS,'{}_{:03d}-of-{:03d}.tfrecord'.format("audios_sources", shard+1, num_shards))
            data_shard = tf_type[shard*im_per_shard:(shard+1)*im_per_shard]
            for data in data_shard:
                with tf.io.TFRecordWriter(output_filename) as writer:
                    for key, value in data.items():
                        save_data(key, value, writer)
        
    
    
    
Get_dataset("./audio")

"""
Boas práticas:
- Salvar o local dos arquivos, ao invés de salvar o arquivo propriamente
- Salvar cada 'linha' processada por vez, para não sobrecarregar na hora de criar o arquivo TFRecords
"""