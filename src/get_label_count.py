from collections import Counter
import pandas as pd

def output_label_count(category_name):
	category_path = f'data/tweeteval/{category_name}/'
	data_files = ['train_labels.txt', 'val_labels.txt', 'test_labels.txt']
	data_names = ['train', 'val', 'test']

	# get label dict
	label_dict = {}
	labels = []
	with open(category_path + 'mapping.txt', 'r') as file:
		for line in file:
			label, emotion = line.strip().split('\t')
			labels.append(label)
			label_dict[int(label)] = emotion

	# print(label_dict, labels)

	df = pd.DataFrame(index=labels, columns=data_names)
	for i in range(3):
		with open(category_path + data_files[i], 'r') as file:
				category_labels = file.read().splitlines()

		label_counts = Counter(category_labels)

		for label in labels:
			df.at[label, data_names[i]] = label_counts[label]
	df.index = list(label_dict.values()) # type: ignore
	# print(df)
	df.to_csv(f'csv_data/tweeteval/{category_name}.csv', index=True)

	# markdown
	markdown_table = df.to_markdown(index=True)
	print('\n' + category_name + '\n')
	print(markdown_table)

def main():
	categories = ["emotion", "hate", "irony", "offensive", "sentiment", "stance/abortion", "stance/atheism", "stance/climate", "stance/feminist", "stance/hillary"]
	for category in categories:
		output_label_count(category)

if __name__ == '__main__':
	main()