import os
import tensorflow as tf
import numpy as np
from Utils import TransformSpec, \
                    audio_to_waveform, \
                    downmix_to_mono, \
                    resample
import torch
from pprint import pprint

transform_spec = TransformSpec()

def _bytes_feature(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()
  elif isinstance(value, str):
    value = value.encode("utf-8")
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

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

def save_data(data, writer):
    tensors = {}
    phase = None
    for index in range(len(data.items())):
        data_waveform, rate_of_sample = audio_to_waveform(data[index])
        data_waveform = downmix_to_mono(data_waveform)
        data_waveform, rate_of_sample = resample(data_waveform, rate_of_sample)
        if index == 0:
            data_spectogram, phase = transform_spec.transform_in_spectogram(data_waveform)
            phase = phase.to(torch.float32)
            phase = tf.io.serialize_tensor(phase.numpy()).numpy()
        else:
            data_spectogram, _ = transform_spec.transform_in_spectogram(data_waveform)
        data_spectogram = data_spectogram.to(torch.float32)
        tensors[index] = tf.io.serialize_tensor(data_spectogram.numpy()).numpy()
    features = {
        "mix": _bytes_feature(tensors[0]),
        "vocals": _bytes_feature(tensors[1]),
        "bass": _bytes_feature(tensors[2]),
        "drums": _bytes_feature(tensors[3]),
        "others": _bytes_feature(tensors[4]),
        "original_phase": _bytes_feature(phase)
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
        
        for shard in range(len(tf_type)):
            output_filename = os.path.join(PATH_DFRECORDS,'{}_{:03d}-of-{:03d}.tfrecord'.format("audios_sources", shard+1, len(tf_type)))
            data_shard = tf_type[shard:shard+1]
            for data in data_shard:
                with tf.io.TFRecordWriter(output_filename) as writer:
                        save_data(data, writer)

Get_dataset("./audio")

"""
Boas práticas:
- Salvar o local dos arquivos, ao invés de salvar o arquivo propriamente
- Salvar cada 'linha' processada por vez, para não sobrecarregar na hora de criar o arquivo TFRecords
"""