import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", help="Path to the dataset folder", required=True)
parser.add_argument("--trainfps", help="FPS of the training video", required=True)
parser.add_argument("--evalfps", help="FPS of the evaluation video", required=True)

args = parser.parse_args()