import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import EncoderDecoderModel

from src.data_loader import load_test_dataset

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class Inference:
    def __init__(self, cfg):
        self.model = self.get_model(cfg.model_name, cfg.model_path)
        self.cfg = cfg
        self.length_penalties = cfg.length_penalty
        self.loader, self.tokenizer = load_test_dataset(cfg.test_path, cfg.model_name)
        self.lp = cfg.length_penalty
        self.save_path = os.path.join(cfg.save_path, cfg.save_name)
        self.submission_path = cfg.submission_path

    def forward(self):
        answers = []
        for batch in tqdm(self.loader):
            batch = {k: v.cuda(device) for k, v in batch.items()}
            generated = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=True,
                bos_token_id=self.tokenizer.cls_token_id,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=0,
                num_beams=5,
                do_sample=False,
                temperature=1.0,
                no_repeat_ngram_size=4,
                bad_words_ids=[[self.tokenizer.unk_token_id]],
                length_penalty=self.lp,
                max_length=512,
            )
            output = self.tokenizer.decode(generated.tolist()[0][1:-1])

            print(output + "\n")
            answers.append(output)

        new_answers = []

        for text in answers:
            new_answers.append(text.replace(' ( ', '(').replace(' ) ', ')').replace(' < ', '<').replace(' > ', '>'))

        submission = pd.read_csv(self.submission_path, index_col=False)
        submission["summary"] = new_answers
        submission.to_csv(self.save_path, sep=',', index=False)

    def get_model(self, model_name: str, model_path: str):
        map_location = torch.device("cpu")
        state_dict = torch.load(
            model_path,
            map_location=map_location
        )

        model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)
        model.load_state_dict(state_dict, strict=False)
        model.cuda(device)
        model.eval()
        return model
