import json
import pandas as pd
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

def create_label2id(mapping_path: Path, out_path: Path):
	print(mapping_path)
	mappings = pd.read_csv(mapping_path, sep='\t', header=None)
	mapping_dict = dict(zip(mappings[1], mappings[0]))
	#print('aa:', mapping_path)
	#print(mapping_dict)
	print('------')
	with open(out_path, 'w', encoding='utf-8') as json_file:
		json.dump(mapping_dict, json_file, ensure_ascii=False, indent=2)


def get_folder_names(directory: Path):
	folder_names = [folder.name for folder in directory.iterdir() if folder.is_dir()]
	return folder_names

def process_folder(input_dir: Path, output_dir: Path, cur_category = None):
	categories = get_folder_names(input_dir)
	if categories:
		for category in categories:
			if not cur_category:
				cur_category = category
			#print(category)
			process_folder(input_dir/category, output_dir/category, cur_category)
	else:
		args = Args().parse_args([])
		args.input_dir = input_dir
		args.output_dir = output_dir
		#print(f"Processing folder: {input_dir.name}")
		#print(f"args: {args}")
		process_category(args, cur_category)

def process_category(args: Args, category):
	print(category)
	data_path = {
		path.stem: path for path in list(args.input_dir.glob("*.txt"))
	}
	
	modes = ["train", "val", "test"]
	attributes = ["text", "labels", "out"]

	mode_list = {
			mode: {attr: f"{mode}_{attr}" for attr in attributes}
			for mode in modes
	}
	mapping_path = data_path.get('mapping')
	if mapping_path:
		# other stance
		out_path = args.output_dir / "label2id.json"
	else:
		# where stance
		mapping_path = args.input_dir.parent / "mapping.txt"
		out_path = args.output_dir / "label2id.json"
	print(mapping_path, out_path)

	if out_path.exists():
		print("Yes")
	else:
		create_label2id(mapping_path, out_path)
		print("No")

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
	process_folder(base_input_dir, base_output_dir, cur_category=None)
