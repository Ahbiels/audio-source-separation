import os
import tensorflow as tf
import numpy as np
from .Utils import TransformSpec, \
                    audio_to_waveform, \
                    downmix_to_mono, \
                    resample
import torch
import os

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
            if track == ".ipynb_checkpoints":
              continue
            print(track_path)
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

def save_data(data, writer, type, new_rate_sample):
    tensors = {}
    chunks = {}
    phase = None
    segment=2 #5 segundos cada chunk

    for index, item in data.items():
        data_waveform, rate_of_sample = audio_to_waveform(item)
        data_waveform = downmix_to_mono(data_waveform)
        # data_waveform, rate_of_sample = resample(data_waveform, rate_of_sample, new_rate_sample)
        tensors[index] = data_waveform
    mix = tensors[0][None]
    chunk_len = int(rate_of_sample * segment)
    length = mix.shape[-1]
    end = chunk_len
    start = 0
    threshold = 0.001
    while start < length:
        chunk_mix = mix[:, :, start:end]

        rms = torch.sqrt(torch.mean(chunk_mix**2))
        if rms < threshold:
            start += chunk_len
            end = start + chunk_len
            continue

        for i, source in tensors.items():
            source_separated = source[None]  # [1, C, T]
            chunk = source_separated[:, :, start:end]

            if chunk.shape[-1] < chunk_len:
                pad_size = chunk_len - chunk.shape[-1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_size))

            chunk = chunk.squeeze(0)
            chunk = chunk.to(torch.float32).cpu().numpy()
            serialized_chunk = tf.io.serialize_tensor(chunk)

            chunks[i] = serialized_chunk


        features = {
            "mix": _bytes_feature(chunks[0]),
            "vocals": _bytes_feature(chunks[1]),
            "bass": _bytes_feature(chunks[2]),
            "drums": _bytes_feature(chunks[3]),
            "others": _bytes_feature(chunks[4]),
        }

        row = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(row.SerializeToString())

        start += chunk_len
        end = start + chunk_len

def Get_dataset(dataset_path, new_rate_sample):
    print("=> Create TFRecords")
    os.system('cls' if os.name == 'nt' else 'clear')
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
                print(f"{shard+1} of {len(tf_type)} - {type}")
                options = tf.io.TFRecordOptions(compression_type="GZIP")
                with tf.io.TFRecordWriter(output_filename, options=options) as writer:
                        save_data(data, writer, type, new_rate_sample)