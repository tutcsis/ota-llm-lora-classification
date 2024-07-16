import json
from pathlib import Path
from tap import Tap
#from tqdm import tqdm

class Args(Tap):
	input_dir: Path = "data/tweeteval/"
	output_dir: Path = "datasets/tweeteval/"
	seed: int = 42

def create_jsonl(text_path, labels_path, output_path, start_id):
	with open(text_path, 'r', encoding='utf-8') as text_file:
		texts = text_file.readlines()

	with open(labels_path, 'r', encoding='utf-8') as labels_file:
		labels = labels_file.readlines()

	with open(output_path, 'w', encoding='utf-8') as output_file:
		for i, (text, label) in enumerate(zip(texts, labels)):
			entry = {
				"id": start_id + i,
				"label": int(label.strip()),
				"title": text.strip()
			}
			json.dump(entry, output_file)
			output_file.write('\n')
	return start_id + len(texts)

def get_folder_names(directory: Path):
	folder_names = [folder.name for folder in directory.iterdir() if folder.is_dir()]
	return folder_names

def process_folder(input_dir: Path, output_dir: Path):
	categories = get_folder_names(input_dir)
	if categories:
		for category in categories:
			process_folder(input_dir/category, output_dir/category)
	else:
		args = Args().parse_args([])
		args.input_dir = input_dir
		args.output_dir = output_dir
		#print(f"Processing folder: {input_dir.name}")
		#print(f"args: {args}")
		process_category(args)

def process_category(args: Args):
	data_path = {
		path.stem: path for path in list(args.input_dir.glob("*.txt"))
	}
	#print(data_path)
	modes = ["train", "val", "test"]
	attributes = ["text", "labels", "out"]

	mode_list = {
			mode: {attr: f"{mode}_{attr}" for attr in attributes}
			for mode in modes
	}
	print(mode_list)

	args.output_dir.mkdir(parents=True, exist_ok=True)

	start_id = 0
	for mode, details in mode_list.items():
		data_path[details["out"]] = args.output_dir / f"{mode}.jsonl"

		start_id = create_jsonl(
			data_path[details["text"]],
			data_path[details["labels"]],
			data_path[details["out"]],
			start_id
		)

if __name__ == '__main__':
	args = Args().parse_args()
	base_input_dir = args.input_dir
	base_output_dir = args.output_dir
	process_folder(base_input_dir, base_output_dir)
