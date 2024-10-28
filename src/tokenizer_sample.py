from transformers import AutoTokenizer
from tap import Tap

import src.utils as utils

class Args(Tap):
	#model_name: str = "rinna/bilingual-gpt-neox-4b"
	model_name: str = "tokyotech-llm/Swallow-7b-hf"
	max_seq_len: int = 512
	seed: int = 42

if __name__ == "__main__":
	args = Args().parse_args()
	utils.init(seed=args.seed)
	print('OK')

	tokenizer = AutoTokenizer.from_pretrained(
		args.model_name,
		model_max_length=args.max_seq_len,
		use_fast=False,
	)

	if tokenizer.pad_token is None:
		tokenizer.add_special_tokens({'pad_token': '[PAD]'})


	print(tokenizer.special_tokens_map)