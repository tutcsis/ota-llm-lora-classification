import pandas as pd
import matplotlib.pyplot as plt

def out_graph(input_paths, output_path, param):
	data_1 = pd.read_csv(input_paths[0])
	data_2 = pd.read_csv(input_paths[1])

	categories = data_1.iloc[:, 0]

	default_fontsize = 16
	param1_values = data_1[param]
	param2_values = data_2[param]
	print(data_1.columns)
	print(data_2.columns)
	print(categories)

	fig, ax = plt.subplots(figsize=(8, 6))
	bar_width = 0.3
	index = range(len(categories))

	bar1 = plt.bar(index, param1_values, bar_width, label='Swallow_13b')
	bar2 = plt.bar([i + bar_width for i in index], param2_values, bar_width, label='llm-jp_13b')

	# plt.xlabel('Category', fontsize=default_fontsize)
	# plt.ylabel('Scores', fontsize=default_fontsize)
	# plt.title(f'Perfoemance of category', fontsize=default_fontsize)
	plt.xticks([i + bar_width/2 for i in index], categories, ha='center', fontsize=default_fontsize)
	plt.yticks(fontsize=default_fontsize)
	plt.legend(loc="upper left", fontsize=default_fontsize)

	plt.tight_layout()
	plt.savefig(output_path, transparent=True)
	plt.close()

if __name__ == '__main__':
  swallow_acc_path = f"csv_data/Swallow_13b_accuracy_loss.csv"
  llmjp_acc_path = f"csv_data/llm-jp_13b_accuracy_loss.csv"
  accuracy_out_path = "imgs/accuracy_graph.png"
  loss_out_path = "imgs/loss_graph.png"

  out_graph(input_paths=[swallow_acc_path, llmjp_acc_path], output_path=loss_out_path, param='test-loss')
  # out_graph(input_paths=[swallow_acc_path, llmjp_acc_path], output_path=accuracy_out_path, param='test-accuracy')
	#out_graph(loss_path, loss_out_path, label='category', param1='val_score', param2='test_score')
	# out_graph(log_path, log_out_path, label='epoch', param1='loss')
