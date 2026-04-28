from workflow import load_checkpoint, UNet, Get_dataset, get_data, train, plot_data, evaluation
import torch
import argparse
import multiprocessing

num_nucleos = multiprocessing.cpu_count()

LEARNING_RATE = 1e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 4
NUM_WORKERS = num_nucleos
NEW_RATE_SAMPLE = 16000
TF_LOCATION_TRAIN="./TFRecords/train/*.tfrecord"
TF_LOCATION_TEST="./TFRecords/test/*.tfrecord"
LOCATION_FILE_MODEL = "./model/my_checkpoint"
FORMAT_FILE_MODEL = ".pth.tar"
AUDIO_LOCATION="./audio"
IN_CHANNELS=1
OUT_CHANNELS=4


def get_args():
    parser = argparse.ArgumentParser(description='Training a UNET model for audio segmentation.')
    parser.add_argument('--batch-size', '-b', dest='batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', '-n', dest='num_epochs', type=int, default=4, help='Num Epochs')
    parser.add_argument('--eval-model', '-e', action='store_true', help='Evaluate model')
    parser.add_argument('--create-dataset', '-c', action='store_true', help='Execute and create TFRecords')
    parser.add_argument('--train', '-t', action='store_true', help='No train the model')
    parser.add_argument('--attempt', '-a', dest='attempt', type=int, default=1, help='training attempt number')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
    load_checkpoint(f"{LOCATION_FILE_MODEL}_{args.attempt-1}{FORMAT_FILE_MODEL}", model)
    
    if args.create_dataset: # -c
        Get_dataset(AUDIO_LOCATION, NEW_RATE_SAMPLE)
    
    if args.train: # -t
        ds_train = get_data(TF_LOCATION_TRAIN, args.batch_size, "train")
        loss_list = train(ds_train, model, args.num_epochs, LEARNING_RATE, f"{LOCATION_FILE_MODEL}_{args.attempt}{FORMAT_FILE_MODEL}", DEVICE)
        plot_data(loss_list, args.attempt, "train")
    
    if args.eval_model: # -e
        ds_test = get_data(TF_LOCATION_TEST, 8, "eval")
        metrics_list = evaluation(ds_test, model, DEVICE)
        plot_data(metrics_list, args.attempt, "eval")
    
    if args.train or args.eval_model:
        while True:
            answer = input("Save the model for inference? (Y/N): ")
            if answer in ("Y", "y", "yes", "Yes", "YES"):
                break
            elif answer in ("N", "n", "no", "No", "NO"):
                break
    print("Pipeline finished")
    