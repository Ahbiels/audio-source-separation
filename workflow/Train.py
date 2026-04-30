import tensorflow as tf
from .Model import UNet
from .Utils import \
                avaliation_model, \
                train_model, \
                save_checkpoint, \
                evaluation_model
# from tqdm import tqdm
# https://apxml.com/courses/getting-started-with-tensorflow/chapter-5-data-input-pipelines-tfdata/working-tfrecord-files
# https://www.tensorflow.org/api_docs/python/tf/data/experimental/parallel_interleave

def parse_data(row_data, purpose):
    base_features = ["mix", "vocals", "bass", "drums", "others"]
    if purpose == "eval":
        base_features.append("original_phase")
    
    feature_description = {
        key: tf.io.FixedLenFeature([], tf.string)
        for key in base_features
    }
    
    parsed = tf.io.parse_single_example(row_data, feature_description)
    
    feature_deserialized = [
        tf.cast(tf.io.parse_tensor(parsed[key], out_type=tf.float32), tf.float32)
        for key in base_features
    ]
    
    return feature_deserialized
    
def split_data(*row_data, purpose):
    X = tf.expand_dims(row_data[0], axis=1)
    y = tf.stack([row_data[1], row_data[2], row_data[3], row_data[4]], axis=-1)
    phase = tf.expand_dims(row_data[5], axis=1) if purpose == "eval" else None
    return X, y, phase

def get_data(path, batch_size, purpose):
    print(f"=> Get {purpose} data")
    ds = tf.data.TFRecordDataset.list_files(path, shuffle=False).interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP"),
        num_parallel_calls=tf.data.AUTOTUNE,
        cycle_length=tf.data.AUTOTUNE
    )
    ds = ds.map(lambda x: parse_data(x, purpose), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda *x: split_data(*x, purpose=purpose), num_parallel_calls=tf.data.AUTOTUNE)
    count = 0
    # ds = ds.cache()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    

    return ds

def train(ds_train, model, num_epochs, learning_rate, model_saved_location, device):
    print("=> Training the model")
    loss_list = []
    count = 0
    for _ in ds_train: #armazenar quantos dados há dentro do dataset
        count += 1
    loss_fn, optimizer = avaliation_model(model, learning_rate)
    for epoch in range(num_epochs):
        for batch_idx, (features, targets, _) in enumerate(ds_train):
            train_model(features, targets, model, loss_fn, optimizer, epoch, batch_idx,loss_list, num_epochs, count, device)
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
        save_checkpoint(checkpoint, model_saved_location)
    
    return loss_list
    
def evaluation(ds_test, model, device):
    print("=> Evaluating the model")
    count = 0
    for _ in ds_test:
        count += 1
    
    return evaluation_model(model, ds_test, count, device)
    
# https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=pt-br#reading_a_tfrecord_file