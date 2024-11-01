from datetime import datetime
from pathlib import Path

import torch
from accelerate import Accelerator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tap import Tap
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AutoTokenizer, BatchEncoding, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer

import src.utils as utils
from src.models import Model

from transformers import file_utils
import os
import json

class Args(Tap):
	def __init__(self, label: str):
		super().__init__()
		print('AAA')
		dataset_dir = Path("./datasets/tweeteval/" + label)
		self.dataset_dir = dataset_dir

	model_name: str = "llm-jp/llm-jp-3-3.7b-instruct"
	#model_name: str = "tokyotech-llm/Swallow-7b-hf"
	model_name: str = "rinna/bilingual-gpt-neox-4b"
	#model_name: str = "rinna/japanese-gpt-neox-3.6b"

	batch_size: int = 32
	epochs: int = 10
	num_warmup_epochs: int = 1

	template_type: int = 2

	lr: float = 5e-4
	lora_r: int = 32
	weight_decay: float = 0.01
	max_seq_len: int = 512
	gradient_checkpointing: bool = True

	seed: int = 42

	def process_args(self):
		self.label2id: dict[str, int] = utils.load_json(self.dataset_dir / "label2id.json")
		self.labels: list[int] = list(self.label2id.values())

		date, time = datetime.now().strftime("%Y-%m-%d/%H-%M-%S.%f").split("/")
		self.output_dir = self.make_output_dir(
			"outputs",
			self.model_name,
			date,
			time,
		)

	def make_output_dir(self, *args) -> Path:
		args = [str(a).replace("/", "__") for a in args]
		output_dir = Path(*args)
		output_dir.mkdir(parents=True)
		return output_dir


class Experiment:
	def __init__(self, args: Args):
		self.args: Args = args

		use_fast = not ("japanese-gpt-neox" in args.model_name)
		self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
			args.model_name,
			model_max_length=args.max_seq_len,
			use_fast=use_fast,
		)

		self.model: PreTrainedModel = Model(
			model_name=args.model_name,
			num_labels=len(args.labels),
			lora_r=args.lora_r,
			gradient_checkpointing=args.gradient_checkpointing,
		).eval()
		self.model.write_trainable_params()
		# llm-jp/llm-jp-3-3.7b-instruct doesn't need token_type_ids
		if "token_type_ids" in self.tokenizer.model_input_names:
			self.tokenizer.model_input_names.remove("token_type_ids")

		self.train_dataloader = self.load_dataset(split="train", shuffle=True)
		steps_per_epoch: int = len(self.train_dataloader)

		self.accelerator = Accelerator()
		(
			self.model,
			self.train_dataloader,
			self.val_dataloader,
			self.test_dataloader,
			self.optimizer,
			self.lr_scheduler,
		) = self.accelerator.prepare(
			self.model,
			self.train_dataloader,
			self.load_dataset(split="val", shuffle=False),
			self.load_dataset(split="test", shuffle=False),
			*self.create_optimizer(steps_per_epoch),
		)

	def load_dataset(
		self,
		split: str,
		shuffle: bool = False,
	) -> DataLoader:
		path: Path = self.args.dataset_dir / f"{split}.jsonl"
		dataset: list[dict] = utils.load_jsonl(path).to_dict(orient="records")
		return self.create_loader(dataset, shuffle=shuffle)

	def build_input(self, title: str) -> str:
		if self.args.template_type == 0:
			return f"tweet: {title}\nlabel: "
		elif self.args.template_type == 1:
			return f"tweet: {title}"
		elif self.args.template_type == 2:
			return f"{title}"

	def collate_fn(self, data_list: list[dict]) -> BatchEncoding:
		title = [d["title"] for d in data_list]
		text = [self.build_input(t) for t in title]


		if self.tokenizer.pad_token is None:
			print(self.tokenizer.special_tokens_map)
			self.tokenizer.pad_token = self.tokenizer.eos_token
			print(self.tokenizer.special_tokens_map)

		inputs: BatchEncoding = self.tokenizer(
			text,
			truncation=True,
			padding=True,
			return_tensors="pt",
			max_length=args.max_seq_len,
		)

		labels = torch.LongTensor([d["label"] for d in data_list])
		return BatchEncoding({**inputs, "labels": labels})

	def create_loader(
		self,
		dataset,
		batch_size=None,
		shuffle=False,
	):
		return DataLoader(
			dataset,
			collate_fn=self.collate_fn,
			batch_size=batch_size or self.args.batch_size,
			shuffle=shuffle,
			num_workers=4,
			pin_memory=True,
		)

	def create_optimizer(
		self,
		steps_per_epoch: int,
	) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
		no_decay = {"bias", "LayerNorm.weight"}
		optimizer_grouped_parameters = [
			{
				"params": [
					param for name, param in self.model.named_parameters() if not name in no_decay
				],
				"weight_decay": self.args.weight_decay,
			},
			{
				"params": [
					param for name, param in self.model.named_parameters() if name in no_decay
				],
				"weight_decay": 0.0,
			},
		]

		optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)

		lr_scheduler = get_linear_schedule_with_warmup(
			optimizer=optimizer,
			num_warmup_steps=steps_per_epoch * self.args.num_warmup_epochs,
			num_training_steps=steps_per_epoch * self.args.epochs,
		)

		return optimizer, lr_scheduler

	def run(self):
		val_metrics = {"epoch": None, **self.evaluate(self.val_dataloader)}
		best_epoch, best_val_f1 = None, val_metrics["f1"]
		best_state_dict = self.model.clone_state_dict()
		self.log(val_metrics)

		loss_epoch_list = []
		for epoch in trange(self.args.epochs, dynamic_ncols=True):
			self.model.train()

			loss_list = []
			for batch in tqdm(
				self.train_dataloader,
				total=len(self.train_dataloader),
				dynamic_ncols=True,
				leave=False,
			):
				self.optimizer.zero_grad()
				out: SequenceClassifierOutput = self.model(**batch)
				loss: torch.FloatTensor = out.loss
				loss_list.append(loss.item())
				self.accelerator.backward(loss)

				self.optimizer.step()
				self.lr_scheduler.step()

			loss_epoch_list.append({'epoch': epoch, 'losses': loss_list})

			self.model.eval()
			val_metrics = {"epoch": epoch, **self.evaluate(self.val_dataloader)}
			self.log(val_metrics)

			if val_metrics["f1"] > best_val_f1:
				best_val_f1 = val_metrics["f1"]
				best_epoch = epoch
				best_state_dict = self.model.clone_state_dict()

		#with open('losses.txt', 'w') as f:
		#	json.dump(loss_epoch_list, f, indent=2)
		#print('output losses collected!!')


		self.model.load_state_dict(best_state_dict)
		self.model.eval()

		val_metrics = {"best-epoch": best_epoch, **self.evaluate(self.val_dataloader)}
		test_metrics = self.evaluate(self.test_dataloader)

		return val_metrics, test_metrics

	@torch.inference_mode()
	def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
		self.model.eval()
		total_loss, gold_labels, pred_labels = 0, [], []

		for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, leave=False):
			out: SequenceClassifierOutput = self.model(**batch)

			batch_size: int = batch.input_ids.size(0)
			loss = out.loss.item() * batch_size
			pred_labels += out.logits.argmax(dim=-1).tolist()
			gold_labels += batch.labels.tolist()
			total_loss += loss

		accuracy: float = accuracy_score(gold_labels, pred_labels)
		precision, recall, f1, _ = precision_recall_fscore_support(
			gold_labels,
			pred_labels,
			average="macro",
			zero_division=0,
			labels=args.labels,
		)

		return {
			"loss": loss / len(dataloader.dataset),
			"accuracy": accuracy,
			"precision": precision,
			"recall": recall,
			"f1": f1,
		}

	def log(self, metrics: dict) -> None:
		utils.log(metrics, self.args.output_dir / "log.csv")
		tqdm.write(
			f"epoch: {metrics['epoch']} \t"
			f"loss: {metrics['loss']:2.6f}   \t"
			f"accuracy: {metrics['accuracy']:.4f} \t"
			f"precision: {metrics['precision']:.4f} \t"
			f"recall: {metrics['recall']:.4f} \t"
			f"f1: {metrics['f1']:.4f}"
		)


def main(args: Args):
	exp = Experiment(args=args)
	print('---- cache place ----')
	print(file_utils.default_cache_path)
	file_utils.default_cache_path = '/work/s245302/.cache/huggingface/'
	print(file_utils.default_cache_path)
	val_metrics, test_metrics = exp.run()

	utils.save_json(val_metrics, args.output_dir / "val-metrics.json")
	utils.save_json(test_metrics, args.output_dir / "test-metrics.json")
	utils.save_config(args, args.output_dir / "config.json")


if __name__ == "__main__":
	#os.environ['HF_HOME'] = 'cache/'
	labels = ["emoji", "emotion", "hate", "irony", "offensive", "sentiment", "stance/abortion", "stance/atheism", "stance/climate", "stance/feminist", "stance/hillary"]
	for label in labels:
		args = Args(label=label).parse_args()
		utils.init(seed=args.seed)
		print('OK')
		main(args)
