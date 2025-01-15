from pathlib import Path
import os
import re
import sys
import json
import pandas as pd
from calc_label_rate import find_folders_in_root, find_subfolders

def output_accuracy_loss_csv(category_paths, category_names, output_name):
	df = pd.DataFrame(index=category_names, columns=['test-accuracy', 'test-loss', 'val-accuracy', 'val-loss'])

	for category_path, category_name in zip(category_paths, category_names):
		print(category_path, category_name)
		with open(category_path + '/test-metrics.json') as f:
			test_metric_data = json.load(f)

		# print(test_metric_data['loss'], test_metric_data['accuracy'])
		df.loc[category_name, 'test-loss'] = round(test_metric_data['loss'], 5)
		df.loc[category_name, 'test-accuracy'] = round(test_metric_data['accuracy'], 5)

		with open(category_path + '/val-metrics.json') as f:
			val_metric_data = json.load(f)

		# print(val_metric_data['loss'], val_metric_data['accuracy'])
		df.loc[category_name, 'val-loss'] = round(val_metric_data['loss'], 5)
		df.loc[category_name, 'val-accuracy'] = round(val_metric_data['accuracy'], 5)

	df = df.sort_index(axis=0)
	print(df)
	df.to_csv(f'csv_data/{output_name}.csv', index=True)

	# markdown
	markdown_table = df.to_markdown(index=True)
	print('\n' + output_name + '\n')
	print(markdown_table)



def main():
	if len(sys.argv) < 5:
		print('Please input 5 variables!!')
		return 0

	dataset_name, model_name, model_option, output_name = sys.argv[1:5]
	# print('dataset: ', dataset_name, 'model: ', model_name, 'model_option: ', model_option, 'output: ', output_name)
	model_path = find_folders_in_root(dataset_name, model_name, option = model_option)
	category_paths, category_names = find_subfolders(model_path)
	
	# print(dataset_name, model_path)
	# print(category_paths)
	# print(category_names)
	output_accuracy_loss_csv(category_paths, category_names, output_name)

if __name__ == "__main__":
	# argv: ['this file path', child_folder, model_name, model_option, output_name]
	main()