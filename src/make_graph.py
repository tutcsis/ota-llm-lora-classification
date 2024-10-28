import pandas as pd
import matplotlib.pyplot as plt

def out_graph(data_path, output_path, label, param1, param2=None):
	data = pd.read_csv(data_path)

	fig, ax = plt.subplots(figsize=(12, 6))
	bar_width = 0.35
	index = range(len(data[label]))

	bar1 = plt.bar(index, data[param1], bar_width, label=param1)
	if param2:
		bar2 = plt.bar([i + bar_width for i in index], data[param2], bar_width, label=param2)

	plt.xlabel(label)
	plt.ylabel('Scores')
	plt.title(f'{label} Perfoemance')
	plt.xticks([i + bar_width/2 for i in index], data[label], rotation=45, ha='right')
	plt.legend()

	plt.tight_layout()

	plt.savefig(output_path)
	plt.close()

if __name__ == '__main__':
	#accuracy_path = "accuracy_list.csv"
	#loss_path = "loss_list.csv"
	#accuracy_out_path = "accuracy_graph.png"
	#loss_out_path = "loss_out.png"
	log_path = "log.csv"
	log_out_path = "llmjp_atheism_log.png"

	#out_graph(accuracy_path, accuracy_out_path, label='category', param1='val_score', param2='test_score')
	#out_graph(loss_path, loss_out_path, label='category', param1='val_score', param2='test_score')
	out_graph(log_path, log_out_path, label='epoch', param1='loss')
