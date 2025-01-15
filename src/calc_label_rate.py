from pathlib import Path
import os
import re
import sys
import json
import pandas as pd

def find_folders_in_root(parent_folder, child_folder, option = ""):
	matching_folders = []

	for name in os.listdir(parent_folder):
		dir_path = os.path.join(parent_folder, name)

		pattern = f"{child_folder}"
		# pattern = f"{child_folder}" if option == "" else f"{child_folder}|{option}"

		if os.path.isdir(dir_path) and re.search(pattern, name):
			matching_folders.append(dir_path)

	if len(matching_folders) > 1:
		print('warning: found more than 2 folders')

	return matching_folders[0]

def find_subfolders(parent_folder):
	subfolder_paths = []
	subfolder_names = []

	for name in os.listdir(parent_folder):
			dir_path = os.path.join(parent_folder, name)
			if os.path.isdir(dir_path):
					subfolder_paths.append(dir_path)
					subfolder_names.append(name)
	return subfolder_paths, subfolder_names


def main():
	if len(sys.argv) < 5:
		print('Please input 5 variables!!')
		return 0

	dataset_name, model_name, model_option, output_dir_path = sys.argv[1:5]
	# print('dataset: ', dataset_name, 'model: ', model_name, 'model_option: ', model_option, 'output: ', output_dir_path)
	model_path = find_folders_in_root(dataset_name, model_name, option = model_option)
	category_paths, category_names = find_subfolders(model_path)
	
	# print(dataset_name, model_path)
	# print(category_paths)
	# print(category_names)
	

if __name__ == "__main__":
	# argv: ['this file path', child_folder, model_name, model_option, output_dir_path]
	main()