import json
import re
from abc import ABC

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class SummaryDataset(Dataset, ABC):
    def __init__(self,
                 texts,
                 summaries=None,
                 tokenizer: BertTokenizer = None,
                 src_max_len: int = 512,
                 gen_max_len: int = 512,
                 ):
        super(SummaryDataset, self).__init__()
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.src_max_len = src_max_len
        self.gen_max_len = gen_max_len

    def __getitem__(self, idx):
        pass

    @staticmethod
    def _squeeze_dict(dic):
        return {key: dic[key].squeeze(0) for key in dic}

    def __len__(self):
        return len(self.texts)


class SummaryTrain(SummaryDataset):
    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]

        src = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.src_max_len,
            truncation=True,
            add_special_tokens=False,
        )

        label = self.tokenizer.encode(summary, add_special_tokens=True, max_length=self.src_max_len,
                                      truncation=True)
        dec_input_ids = label

        src = self._squeeze_dict(src)
        src.update({'decoder_input_ids': torch.LongTensor(dec_input_ids)})
        src.update({'decoder_attention_masks': torch.LongTensor([1] * len(label))})
        src.update({'labels': torch.LongTensor(label)})

        return src


class SummaryTest(SummaryDataset):
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.src_max_len:
            sents = text.split('. ')
            prev_src = self.tokenizer(
                '. '.join(sents[:20]),
                return_tensors='pt',
                max_length=self.src_max_len,
                truncation=True,
                add_special_tokens=False,
            )
            prev_src = self._squeeze_dict(prev_src)
            post_src = self.tokenizer(
                '. '.join(sents[-4:]),
                return_tensors='pt',
                max_length=self.src_max_len,
                truncation=True,
                add_special_tokens=False,
            )
            post_src = self._squeeze_dict(post_src)
            if len(prev_src['input_ids']) + len(post_src['input_ids']) > self.src_max_len:
                prev_src = {key: value[:self.src_max_len - len(post_src['input_ids'])] for key, value in
                            prev_src.items()}
            src = {}
            for key in prev_src:
                src[key] = torch.cat([prev_src[key], post_src[key]], dim=0)
        else:
            src = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=self.src_max_len,
                truncation=True,
                add_special_tokens=False,
            )

            src = self._squeeze_dict(src)
        return src

    def preprocess_str(self, string):
        string = re.sub('[ㄱ-ㅎ가-힣]{1,5} 기자[^가-힣\[\]\(\)\{\}]', '', string)  # 'XXX 기자' 제거
        string = re.sub('[\w\.-]+@[\w\.-]+\.\w+', '', string)  # 이메일 제거
        string = ' '.join([s for s in string.split() if len(re.findall('[가-힣a-z0-9]+', s)) > 0])  # 특수문자 단어
        string = ' '.join([re.sub('\(.*\)|\{.*\}|\[.*\]', '', s) for s in string.split()])  # 괄호 패턴 제거
        string = re.sub('\(.{0,15}\)|\{.{0,15}\}|\[.{0,15}\]', '', string)  # 괄호 패턴 제거2

        string = re.sub(' +', ' ', string).strip()  # 중복 공백 & 양 끝 공백 제거
        return string


def make_batch_train(samples):
    """
    indices로 train data 의 batch를 묶을 때 필요한 함수
    Args:
        samples:

    Returns:

    """
    batch_dict = {key: list() for key in samples[0]}
    for inp in samples:
        for key in inp:
            batch_dict[key].append(inp[key])

    batch = dict()
    for key in batch_dict:
        batch[key] = pad_sequence(batch_dict[key]).transpose(1, 0)

    return batch


def make_batch_test(samples):
    """
    indices로 test data의 batch를 묶을 때 필요한 함수
    Args:
        samples:

    Returns:

    """
    batch_dict = {key: list() for key in samples[0]}
    for inp in samples:
        for key in inp:
            batch_dict[key].append(inp[key])

    batch = dict()
    for key in batch_dict:
        if key == 'labels':
            batch[key] = pad_sequence(batch_dict[key], padding_value=-100).transpose(1, 0)

        else:
            batch[key] = pad_sequence(batch_dict[key]).transpose(1, 0)

    return batch


def load_train_dataset(train_path: str, model_name: str, batch_size: int, num_workers: int):
    with open(train_path, 'r') as f:
        train_data = json.loads(f.read())

    train_texts = []
    train_sums = []

    for data in train_data:
        for agendas in data["context"]:
            train_sums.append(data['label'][agendas]['summary'])
            texts = []
            for agenda in data["context"][agendas].values():
                texts.append(agenda)

            texts = ' '.join(texts)
            train_texts.append(texts)

    enc_tokenizer = BertTokenizer.from_pretrained(model_name)
    dataset = SummaryTrain(texts=train_texts, summaries=train_sums, tokenizer=enc_tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                             num_workers=num_workers, collate_fn=make_batch_train)
    return data_loader


def load_test_dataset(test_path: str, model_name: str):
    with open(test_path, 'r') as f:
        test_data = json.loads(f.read())

    test_tests = []
    for data in test_data:
        for agendas in data["context"]:
            texts = []
            for agenda in data["context"][agendas].values():
                texts.append(agenda)
            texts = ' '.join(texts)
            test_tests.append(texts)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    dataset = SummaryTest(texts=test_tests, tokenizer=tokenizer)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True,
                             num_workers=1, collate_fn=make_batch_train)
    return data_loader, tokenizer

