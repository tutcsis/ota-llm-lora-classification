import pandas as pd
from pathlib import Path
from tap import Tap
from collections import Counter
import csv


class Args(Tap):
	input_dir: Path = "data/tweeteval/stance/atheism/"

def read_labels(label_path):
	with open(label_path, "r") as f:
		data = f.readlines()
	return [int(line.strip()) for line in data]

def main(args: Path):
	train_path = args.input_dir / "train_labels.txt"
	val_path = args.input_dir / "val_labels.txt"
	test_path = args.input_dir / "test_labels.txt"
	output_path = "output_ratio.csv"

	train_labels = read_labels(train_path)
	val_labels = read_labels(val_path)
	test_labels = read_labels(test_path)



	train_counts = Counter(train_labels)
	val_counts = Counter(val_labels)
	test_counts = Counter(test_labels)

	total_train = sum(train_counts.values())
	total_val = sum(val_counts.values())
	total_test = sum(test_counts.values())

	all_labels = sorted(set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys()))
	print(all_labels)

	train_ratios = {label: train_counts.get(label, 0) / total_train if total_train > 0 else 0 for label in all_labels}
	val_ratios = {label: val_counts.get(label, 0) / total_val if total_val > 0 else 0 for label in all_labels}
	test_ratios = {label: test_counts.get(label, 0) / total_test if total_test > 0 else 0 for label in all_labels}

	df = pd.DataFrame([train_ratios, val_ratios, test_ratios], index=['train', 'val', 'test']).reset_index()
	df = df.rename(columns={'index': ''})

	df.to_csv(output_path, index=False)


if __name__ == '__main__':
	args = Args().parse_args()
	main(args)

