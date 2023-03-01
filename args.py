import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for packed data format benchmark"
    )

    # Dataset choice
    datasets = ["CIFAR10", "ImageNet10k", "FFHQ"]
    parser.add_argument(
        "-d", "--dataset", type=str, default="CIFAR10", choices=datasets
    )

    # Choice of image formats
    formats = ["Image", "HDF5", "LMDB", "Petastorm", "TAR", "TFRecords", "ZIP"]
    parser.add_argument("-f", "--format", nargs="+", default=formats, choices=formats)
    parser.add_argument("-le", "--load-encoded", type=int, default=0, choices=[0, 1])

    # If data is copied to tmp or dev
    parser.add_argument(
        "-l",
        "--location",
        type=str,
        default="home",
        choices=["home", "/scratch-local/", "/scratch-shared/"],
    )

    # If data is cached within the dataset class
    parser.add_argument("-c", "--cache", type=int, default=0, choices=[0, 1])

    # Dataloader specific
    parser.add_argument("-ws", "--warm-start", type=int, default=1, choices=[0, 1])
    parser.add_argument("-pm", "--pin-memory", type=int, default=1, choices=[0, 1])
    parser.add_argument(
        "-pw", "--persistent-workers", type=int, default=0, choices=[0, 1]
    )
    parser.add_argument("-e", "--epochs", type=int, default=2)
    parser.add_argument("-bs", "--batch-size", type=int, default=16)
    # cpu_count = len(os.sched_getaffinity(0))
    # num_workers = [0] + [2**i for i in range(int(np.log(cpu_count) / np.log(2)))]
    parser.add_argument("-nw", "--num_workers", type=int, nargs="+", default=[32])
    parser.add_argument("-t", "--transform", type=int, default=1, choices=[0, 1])
    parser.add_argument("-s", "--shuffle", type=int, default=[1], choices=[0, 1])

    # Device
    parser.add_argument(
        "-dvc", "--device", type=str, default="cuda", choices=["cpu", "cuda"]
    )

    return parser.parse_args()
