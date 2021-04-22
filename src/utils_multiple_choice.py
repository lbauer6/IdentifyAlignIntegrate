# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Notice: Majority of code borrowed from https://github.com/huggingface/transformers
# Main changes made to data input pipeline (new processors and featurizers). 


""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """


import csv
import glob
import json
import logging
import os
from typing import List

import tqdm

from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, endings, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label


class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids}
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class PiqaBaselineProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir, balanced=False):
        """See base class."""
        lines = []
        with open(os.path.join(data_dir, "train.jsonl")) as json_file:
            for f in json_file:
                lines.append(json.loads(f))
        if not balanced: 
            return self._create_examples(lines, "train")
        else:
            return self._create_balanced_examples(lines, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines = []
        with open(os.path.join(data_dir, "dev.jsonl")) as json_file:
            for f in json_file:
                lines.append(json.loads(f))
 
        return self._create_examples(lines, "dev")


    def get_test_examples(self, data_dir):
        """See base class."""
        lines = []
        with open(os.path.join(data_dir, "dev.jsonl")) as json_file:
            for f in json_file:
                lines.append(json.loads(f))
 
        return self._create_examples(lines, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        examples = []
        lines  = chunks(lines, 2)
        for (i, d_i) in enumerate(lines):
            id_ = "%s-%s" % (type, i)
            question_ = "" 
            context_= [d_i[0]["sentence1"], d_i[0]["sentence1"]]
            endings_ = []
            num_cs = 0 
            for j, line in enumerate(d_i):
                end = line["sentence2"]
                if line["sentence2"] == line["gold_label"]: 
                    label_ = str(j) 
                    if line["pos_commonsense"]:
                        num_cs += 1 
                elif line["neg_commonsense"]:
                    num_cs += 1 
                endings_.append(end)
            if num_cs!=2:
                continue 
            examples.append(
                     InputExample(example_id=id_, \
                             question=question_, \
                             endings=endings_,\
                             contexts=context_,\
                             label=label_))
        return examples

    def _create_balanced_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        pos_examples = []
        neg_examples = []
        lines  = chunks(lines, 2)
        for (i, d_i) in enumerate(lines):
            id_ = "%s-%s" % (type, i)
            question_ = "" 
            context_= [d_i[0]["sentence1"], d_i[0]["sentence1"]]
            endings_ = []
            num_cs = 0 
            neg = False
            pos = False
            for j, line in enumerate(d_i):
                end = line["sentence2"]
                if line["sentence2"] == line["gold_label"]: 
                    label_ = str(j) 
                    if line["pos_commonsense"]:
                        pos = True
                        num_cs += 1 
                elif line["neg_commonsense"]:
                    neg = True
                    num_cs += 1 
                endings_.append(end)
            if num_cs!=2:
                continue
            if neg: 
                neg_examples.append(
                         InputExample(example_id=id_, \
                                 question=question_, \
                                 endings=endings_,\
                                 contexts=context_,\
                                 label=label_))
            elif pos: 
                pos_examples.append(
                         InputExample(example_id=id_, \
                                 question=question_, \
                                 endings=endings_,\
                                 contexts=context_,\
                                 label=label_))
        if len(neg_examples) > len(pos_examples):
            neg_examples = neg_examples[:len(pos_examples)] 
        else:
            pos_examples = pos_examples[:len(neg_examples)] 
        #print ("neg data: ",len(neg_examples)) 
        #print ("pos data: ",len(pos_examples)) 
        examples = pos_examples + neg_examples  
        #print ("data: ", len(examples)) 
        return examples

class PiqaWHKSProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir, balanced=False):
        """See base class."""
        lines = []
        with open(os.path.join(data_dir, "train.jsonl")) as json_file:
            for f in json_file:
                lines.append(json.loads(f))
        if not balanced: 
            return self._create_examples(lines, "train")
        else:
            return self._create_balanced_examples(lines, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines = []
        with open(os.path.join(data_dir, "dev.jsonl")) as json_file:
            for f in json_file:
                lines.append(json.loads(f))
 
        return self._create_examples(lines, "dev")


    def get_test_examples(self, data_dir):
        """See base class."""
        lines = []
        with open(os.path.join(data_dir, "dev.jsonl")) as json_file:
            for f in json_file:
                lines.append(json.loads(f))
 
        return self._create_examples(lines, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

                         
    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        examples = []
        lines  = chunks(lines, 2)
        for (i, d_i) in enumerate(lines):
            id_ = "%s-%s" % (type, i)
            question_ = "" 
            context_= [d_i[0]["sentence1"], d_i[0]["sentence1"]]
            endings_ = []
            num_cs = 0 
            for j, line in enumerate(d_i):
                end = line["sentence2"]
                if line["sentence2"] == line["gold_label"]: 
                    label_ = str(j) 
                    if line["pos_commonsense"]:
                        end += " . " + line["pos_commonsense"]
                        num_cs += 1 
                elif line["neg_commonsense"]:
                    end += " . " + line["neg_commonsense"]
                    num_cs += 1 
                endings_.append(end)
            if num_cs!=2:
                continue 
            examples.append(
                     InputExample(example_id=id_, \
                             question=question_, \
                             endings=endings_,\
                             contexts=context_,\
                             label=label_))
        return examples




class PiqaKSProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir, balanced=False):
        """See base class."""
        lines = []
        with open(os.path.join(data_dir, "train.jsonl")) as json_file:
            for f in json_file:
                lines.append(json.loads(f))
        if not balanced: 
            return self._create_examples(lines, "train")
        else:
            return self._create_balanced_examples(lines, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines = []
        with open(os.path.join(data_dir, "dev.jsonl")) as json_file:
            for f in json_file:
                lines.append(json.loads(f))
 
        return self._create_examples(lines, "dev")


    def get_test_examples(self, data_dir):
        """See base class."""
        lines = []
        with open(os.path.join(data_dir, "dev.jsonl")) as json_file:
            for f in json_file:
                lines.append(json.loads(f))
 
        return self._create_examples(lines, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_balanced_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        pos_examples = []
        neg_examples = []
        lines  = chunks(lines, 2)
        for (i, d_i) in enumerate(lines):
            id_ = "%s-%s" % (type, i)
            question_ = "" 
            context_= [d_i[0]["sentence1"], d_i[0]["sentence1"]]
            endings_ = []
            num_cs = 0 
            neg = False
            pos = False
            for j, line in enumerate(d_i):
                end = line["sentence2"]
                if line["sentence2"] == line["gold_label"]: 
                    label_ = str(j) 
                    if line["pos_commonsense"]:
                        pos = True
                        end += " . " + " . ".join(line["pos_commonsense"])
                        num_cs += 1 
                elif line["neg_commonsense"]:
                    neg = True
                    end += " . " + " . ".join(line["neg_commonsense"])
                    num_cs += 1 
                endings_.append(end)
            if num_cs!=2:
                continue
            if neg: 
                neg_examples.append(
                         InputExample(example_id=id_, \
                                 question=question_, \
                                 endings=endings_,\
                                 contexts=context_,\
                                 label=label_))
            elif pos: 
                pos_examples.append(
                         InputExample(example_id=id_, \
                                 question=question_, \
                                 endings=endings_,\
                                 contexts=context_,\
                                 label=label_))
        if len(neg_examples) > len(pos_examples):
            neg_examples = neg_examples[:len(pos_examples)] 
        else:
            pos_examples = pos_examples[:len(neg_examples)] 
        #print ("neg data: ",len(neg_examples)) 
        #print ("pos data: ",len(pos_examples)) 
        examples = pos_examples + neg_examples  
        #print ("data: ", len(examples)) 
        return examples


    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        examples = []
        raw_ex = [] 
        lines  = chunks(lines, 2)
        for (i, d_i) in enumerate(lines):
            id_ = "%s-%s" % (type, i)
            question_ = "" 
            context_= [d_i[0]["sentence1"], d_i[0]["sentence1"]]
            endings_ = []
            num_cs = 0 
            for j, line in enumerate(d_i):
                end = line["sentence2"]
                if line["sentence2"] == line["gold_label"]: 
                    label_ = str(j) 
                    if line["pos_commonsense"]:
                        end += " . " + " . ".join(line["pos_commonsense"])
                        num_cs += 1 
                elif line["neg_commonsense"]:
                    end += " . " + " . ".join(line["neg_commonsense"])
                    num_cs += 1 
                endings_.append(end)
            if num_cs!=2:
                continue 
            raw_ex.append(d_i) 
 
            examples.append(
                     InputExample(example_id=id_, \
                             question=question_, \
                             endings=endings_,\
                             contexts=context_,\
                             label=label_))
        print_data(raw_ex[:10]) 
        return examples

def print_data(examples):
    for d in examples:
        d_i = d[0] 
        print ("prompt", " : ", d_i["sentence1"])
        print ("gold", " : ", d_i["gold_label"])
        print ("pos_cs", " : ", d_i["pos_commonsense"])
        for d_j in d:
            if d_j["sentence2"] != d_i["gold_label"]: 
                print ("neg", " : ", d_j["sentence2"])
                print ("neg cs", " : ", d_j["neg_commonsense"])
        print ("--------")


class MCKSProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        lines = []
        with open(os.path.join(data_dir, "train.jsonl")) as json_file:
            for f in json_file:
                lines.append(json.loads(f))
 
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines = []
        with open(os.path.join(data_dir, "dev.jsonl")) as json_file:
            for f in json_file: lines.append(json.loads(f))
 
        return self._create_examples(lines, "dev")


    def get_test_examples(self, data_dir):
        """See base class."""
        lines = []
        with open(os.path.join(data_dir, "dev.jsonl")) as json_file:
            for f in json_file:
                lines.append(json.loads(f))
 
        return self._create_examples(lines, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        examples = []
        raw_ex = [] 
        lines  = chunks(lines, 2)
        for (i, d_i) in enumerate(lines):
            id_ = "%s-%s" % (type, i)
            question_ = "" 
            context_= [d_i[0]["sentence1"], d_i[0]["sentence1"]]
            endings_ = []
            num_cs = 0 
            for j, line in enumerate(d_i):
                end = line["sentence2"]
                if line["sentence2"] == line["gold_label"]: 
                    label_ = str(j) 
                    if line["pos_commonsense"]:
                        num_cs += 1 
                elif line["neg_commonsense"]:
                    num_cs += 1 
                endings_.append(end)
            if num_cs!=2:
                continue 
            raw_ex.append(d_i) 
            examples.append(
                     InputExample(example_id=id_, \
                             question=question_, \
                             endings=endings_,\
                             contexts=context_,\
                             label=label_))
        return examples

class MCBaselineProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = []
        with open(os.path.join(data_dir, "train.jsonl")) as json_file:
            for f in json_file:
                lines.append(json.loads(f))
 
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines = []
        with open(os.path.join(data_dir, "dev.jsonl")) as json_file:
            for f in json_file: lines.append(json.loads(f))
 
        return self._create_examples(lines, "dev")


    def get_test_examples(self, data_dir):
        """See base class."""
        lines = []
        with open(os.path.join(data_dir, "dev.jsonl")) as json_file:
            for f in json_file:
                lines.append(json.loads(f))
 
        return self._create_examples(lines, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        examples = []
        lines  = chunks(lines, 2)
        for (i, d_i) in enumerate(lines):
            id_ = "%s-%s" % (type, i)
            question_ = "" 
            context_= [d_i[0]["sentence1"], d_i[0]["sentence1"]]
            endings_ = []
            num_cs = 0 
            for j, line in enumerate(d_i):
                end = line["sentence2"]
                if line["sentence2"] == line["gold_label"]: 
                    label_ = str(j) 
                    if line["pos_commonsense"]:
                        num_cs += 1 
                elif line["neg_commonsense"]:
                    num_cs += 1 
                endings_.append(end)
            if num_cs!=2:
                continue 
            examples.append(
                     InputExample(example_id=id_, \
                             question=question_, \
                             endings=endings_,\
                             contexts=context_,\
                             label=label_))
        return examples

 
class MCScriptProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = []
        with open(os.path.join(data_dir, "train.jsonl")) as json_file:
            for f in json_file:
                lines.append(json.loads(f))
 
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines = []
        with open(os.path.join(data_dir, "dev.jsonl")) as json_file:
            for f in json_file: lines.append(json.loads(f))
 
        return self._create_examples(lines, "dev")


    def get_test_examples(self, data_dir):
        """See base class."""
        lines = []
        with open(os.path.join(data_dir, "dev.jsonl")) as json_file:
            for f in json_file:
                lines.append(json.loads(f))
 
        return self._create_examples(lines, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        lines  = chunks(lines, 2)
        examples = []
        for (i, d_i) in enumerate(lines):
            id_ = "%s-%s" % (type, i)
            question_ = "" 
            context_= [d_i[0]["sentence1"], d_i[0]["sentence1"]]
            endings_ = []
            for j, line in enumerate(d_i):
                end = line["sentence2"]
                if line["sentence2"] == line["gold_label"]: 
                    label_ = str(j) 
                endings_.append(end)
            examples.append(
                     InputExample(example_id=id_, \
                             question=question_, \
                             endings=endings_,\
                             contexts=context_,\
                             label=label_))
        return examples



class PiqaProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = []
        with open(os.path.join(data_dir, "train.jsonl")) as json_file:
            for f in json_file:
                lines.append(json.loads(f))
 
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines = []
        with open(os.path.join(data_dir, "dev.jsonl")) as json_file:
            for f in json_file: lines.append(json.loads(f))
 
        return self._create_examples(lines, "dev")


    def get_test_examples(self, data_dir):
        """See base class."""
        lines = []
        with open(os.path.join(data_dir, "dev.jsonl")) as json_file:
            for f in json_file:
                lines.append(json.loads(f))
 
        return self._create_examples(lines, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            id_ = "%s-%s" % (type, i)
            question_ = "" 
            endings_= [line["sol1"], line["sol2"]]
            context_= [line["goal"], line["goal"]]
            label_ = str(line["label"]) 
            examples.append(
                     InputExample(example_id=id_, \
                             question=question_, \
                             endings=endings_,\
                             contexts=context_,\
                             label=label_))
        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending
            inputs = tokenizer.encode_plus(text_a, text_b, add_special_tokens=True, max_length=max_length,)
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            choices_features.append((input_ids, attention_mask, token_type_ids))

        label = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("_id: {}".format(example.example_id))
            for choice_idx, (input_ids, attention_mask, token_type_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
                logger.info("attention_mask: {}".format(" ".join(map(str, attention_mask))))
                logger.info("token_type_ids: {}".format(" ".join(map(str, token_type_ids))))
                logger.info("label: {}".format(label))
        features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label,))
    return features


processors = {"piqa": PiqaProcessor, "piqaks": PiqaKSProcessor,\
                    "piqabaseline": PiqaBaselineProcessor, "piqawhks": PiqaWHKSProcessor, "mc2": MCScriptProcessor, "mc2baseline": MCBaselineProcessor, "mc2ks": MCKSProcessor}


MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"piqa", 2, "piqaks", 2, "piqabaseline", 2, "piqawhks", 2, "mc2", 2, "mcbaseline"}

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
