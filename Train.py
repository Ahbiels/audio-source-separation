import argparse
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
                evaluation_model, \
                plot_data
# from tqdm import tqdm
import os
# https://apxml.com/courses/getting-started-with-tensorflow/chapter-5-data-input-pipelines-tfdata/working-tfrecord-files
# https://www.tensorflow.org/api_docs/python/tf/data/experimental/parallel_interleave

def get_args():
    parser = argparse.ArgumentParser(description='Training a UNET model for audio segmentation.')
    parser.add_argument('--batch-size', '-b', dest='batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--purpose', '-p', dest='purpose', type=str, default="train", help='Purpose (train or eval)')
    parser.add_argument('--attempt', '-a', dest='attempt', type=int, default=1, help='training attempt number')
    
    return parser.parse_args()

def parse_data(row_data, purpose):
    feature_description = {
        'mix': tf.io.FixedLenFeature([], tf.string),
        'vocals': tf.io.FixedLenFeature([], tf.string),
        'bass': tf.io.FixedLenFeature([], tf.string),
        'drums': tf.io.FixedLenFeature([], tf.string),
        'others': tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(row_data, feature_description)
    mix_deserialized = tf.cast(tf.io.parse_tensor(parsed["mix"], out_type=tf.float32), tf.float32)
    vocals_deserialized = tf.cast(tf.io.parse_tensor(parsed["vocals"], out_type=tf.float32), tf.float32)
    bass_deserialized = tf.cast(tf.io.parse_tensor(parsed["bass"], out_type=tf.float32), tf.float32)
    drums_deserialized = tf.cast(tf.io.parse_tensor(parsed["drums"], out_type=tf.float32), tf.float32)
    others_deserialized = tf.cast(tf.io.parse_tensor(parsed["others"], out_type=tf.float32), tf.float32)
    
    if purpose == "eval":
        feature_description["original_phase"] = tf.io.FixedLenFeature([], tf.string)
        phase_deserialized = tf.cast(tf.io.parse_tensor(parsed["original_phase"], out_type=tf.float32), tf.float32)
        return mix_deserialized, vocals_deserialized, bass_deserialized, drums_deserialized, others_deserialized, phase_deserialized
    else:
        return mix_deserialized, vocals_deserialized, bass_deserialized, drums_deserialized, others_deserialized
    
def split_data(*row_data, purpose):
    X = tf.expand_dims(row_data[0], axis=1)
    y = tf.stack([row_data[1], row_data[2], row_data[3], row_data[4]], axis=-1)
    phase = tf.expand_dims(row_data[5], axis=1) if purpose == "eval" else None
    return X, y, phase

def get_data(path, batch_size, purpose):
    ds = tf.data.TFRecordDataset.list_files(path, shuffle=False).interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=tf.data.AUTOTUNE,
        cycle_length=tf.data.AUTOTUNE
    )
    ds = ds.map(lambda x: parse_data(x, purpose), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda *x: split_data(*x, purpose=purpose), num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = ds.cache()
    
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds

def train(batch_size, purpose, attempt):
    count = 0
    loss_list = []
    ds_train = get_data(TF_LOCATION, batch_size, purpose)
    # armazenar quantos dados há dentro do dataset
    for _ in ds_train:
        count += 1
    model = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
    if os.path.isfile(MODEL_LOCATION):
        load_checkpoint(MODEL_LOCATION, model)
    loss_fn, optimizer = avaliation_model(model)
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (features, targets, _) in enumerate(ds_train):
            train_model(features, targets, model, loss_fn, optimizer, epoch, batch_idx,loss_list, count)
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
        save_checkpoint(checkpoint)
    
    plot_data(loss_list, attempt)
    
def evaluation():
    ds_test = get_data(TF_LOCATION)

    model = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
    load_checkpoint(MODEL_LOCATION, model)
    
    evaluation_model(model, ds_test)

if __name__ == "__main__":
    args = get_args()
    os.system('cls' if os.name == 'nt' else 'clear')

    if args.purpose == "train":
        train(args.batch_size, args.purpose, args.attempt)
    elif args.purpose == "eval":
        evaluation()
    else:
        print(f"{args.purpose} is invalid. Please, use 'train' or 'eval'")
    
# https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=pt-br#reading_a_tfrecord_file