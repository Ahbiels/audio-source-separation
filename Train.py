import tensorflow as tf
from Model import UNet
from Utils import \
                DEVICE, \
                NUM_EPOCHS, \
                TF_LOCATION, \
                MODEL_LOCATION, \
                IN_CHANNELS, \
                OUT_CHANNELS, \
                avaliation_model, \
                train_model, \
                save_checkpoint, \
                load_checkpoint, \
                evaluation_model
# from tqdm import tqdm
import os.path
# https://apxml.com/courses/getting-started-with-tensorflow/chapter-5-data-input-pipelines-tfdata/working-tfrecord-files
# https://www.tensorflow.org/api_docs/python/tf/data/experimental/parallel_interleave

def parse_data(row_data):
    feature_description = {
        'mix': tf.io.FixedLenFeature([], tf.string),
        'vocals': tf.io.FixedLenFeature([], tf.string),
        'bass': tf.io.FixedLenFeature([], tf.string),
        'drums': tf.io.FixedLenFeature([], tf.string),
        'others': tf.io.FixedLenFeature([], tf.string),
        'original_phase': tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(row_data, feature_description)
    mix_deserialized = tf.cast(tf.io.parse_tensor(parsed["mix"], out_type=tf.float32), tf.float32)
    vocals_deserialized = tf.cast(tf.io.parse_tensor(parsed["vocals"], out_type=tf.float32), tf.float32)
    bass_deserialized = tf.cast(tf.io.parse_tensor(parsed["bass"], out_type=tf.float32), tf.float32)
    drums_deserialized = tf.cast(tf.io.parse_tensor(parsed["drums"], out_type=tf.float32), tf.float32)
    others_deserialized = tf.cast(tf.io.parse_tensor(parsed["others"], out_type=tf.float32), tf.float32)
    phase_deserialized = tf.cast(tf.io.parse_tensor(parsed["original_phase"], out_type=tf.float32), tf.float32)
    
    return mix_deserialized, vocals_deserialized, bass_deserialized, drums_deserialized, others_deserialized, phase_deserialized
    
def split_data(*row_data):
    X = tf.expand_dims(row_data[0], axis=1)
    y = tf.stack([row_data[1], row_data[2], row_data[3], row_data[4]], axis=-1)
    phase = tf.expand_dims(row_data[5], axis=1)
    return X, y, phase

def get_data(path):
    ds = tf.data.TFRecordDataset.list_files(path, shuffle=False).interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=tf.data.AUTOTUNE,
        cycle_length=tf.data.AUTOTUNE
    )
    ds = ds.map(parse_data, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(split_data, num_parallel_calls=tf.data.AUTOTUNE)
    # ds = ds.cache()
    
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds

def train():
    ds_train = get_data(TF_LOCATION)
    model = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
    # if os.path.isfile(MODEL_LOCATION):
    #     load_checkpoint(MODEL_LOCATION, model)
    loss_fn, optimizer = avaliation_model(model)
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (features, targets, _) in enumerate(ds_train):
            train_model(features, targets, model, loss_fn, optimizer, epoch, batch_idx)
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
        save_checkpoint(checkpoint)
    
def evaluation():
    ds_test = get_data(TF_LOCATION)

    model = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
    load_checkpoint(MODEL_LOCATION, model)
    
    evaluation_model(model, ds_test)
    
train()
        

# evaluation()
# https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=pt-br#reading_a_tfrecord_file