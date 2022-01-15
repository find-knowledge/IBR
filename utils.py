from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open

import json
from nltk.tokenize import sent_tokenize
import numpy as np

from proof_utils import get_proof_graph, get_proof_graph_with_fail

logger = logging.getLogger(__name__)

# RRInputExample(id, context, question, parent_node, child_node, decoded_node_pairs)
class RRInputExample(object):
    def __init__(self, id, context, question, parent_node, child_node, decoded_node_pairs, decoded_node, label):
        self.id = id
        self.context = context
        self.question = question
        self.parent_node = parent_node
        self.child_node = child_node
        self.decoded_node_pairs = decoded_node_pairs
        self.decoded_node = decoded_node
        self.label = label

class RRFeatures(object):
    def __init__(self, id, input_ids, input_mask, segment_ids, proof_offset, node_label, edge_label, attention_node, label_id):
        self.id = id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.proof_offset = proof_offset
        # self.node_label = node_label
        # self.edge_label = edge_label
        self.attention_node = attention_node
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

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

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_jsonl(cls, input_file):
        """Reads a tab separated value file."""
        records = []
        with open(input_file, "r", encoding="utf-8-sig") as f:
            for line in f:
                records.append(json.loads(line))
            return records




class RRProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train.jsonl")),
            self._read_jsonl(os.path.join(data_dir, "meta-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        # Change these to test paths for test results
        self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "dev.jsonl")),
            self._read_jsonl(os.path.join(data_dir, "meta-dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "test.jsonl")),
            self._read_jsonl(os.path.join(data_dir, "meta-test.jsonl")), "test")

    def get_labels(self):
        return [True, False]

    # Unconstrained training, use this for ablation
    def _get_node_labels(self, proofs, sentence_scramble, nfact, nrule):
        proof = proofs.split("OR")[0]

        node_label = [0] * (nfact + nrule + 1)
        edge_label = np.zeros((nfact + nrule + 1, nfact + nrule + 1), dtype=int)
        FAIL_flag = 0
        if "FAIL" in proof:
            nodes, edges = get_proof_graph_with_fail(proof)
            FAIL_flag = 1

        else:
            nodes, edges = get_proof_graph(proof)

        component_index_map = {}
        for (i, index) in enumerate(sentence_scramble):
            if index <= nfact:
                component = "triple" + str(index)
            else:
                component = "rule" + str(index-nfact)
            component_index_map[component] = i
            # print(component_index_map)

        for node in nodes:
            if node != "NAF":
                index = component_index_map[node]
            else:
                index = nfact+nrule
            node_label[index] = 1

        node_pairs = []
        for index, edge in enumerate(edges):
            if edge[0] != "NAF":
                start_index = component_index_map[edge[0]]
            else:
                start_index = nfact + nrule
            if edge[1] != "NAF":
                end_index = component_index_map[edge[1]]
            else:
                end_index = nfact + nrule

            edge_label[end_index][start_index] = 1

            if index == 0:
                node_pairs.append([start_index, len(node_label)])
            if index == len(edges) - 1:
                node_pairs.append([len(node_label), end_index])
            node_pairs.append([end_index, start_index])


        if len(nodes) == 1:
            node_pairs.append([len(node_label), node_label.index(1)])
            node_pairs.append([node_label.index(1), len(node_label)])
            # node_pairs.append(node_label.index(1))
        else:
            node_pairs = list(reversed(node_pairs))

        # node_label, list(edge_label.flatten())
        return node_pairs, node_label

    def _create_examples(self, records, meta_records, data_type):
        output_file = open("./data/new_data/"+data_type+".csv", 'w', encoding='utf-8')
        output_file.write("id\tcontext\tquestion\ttarget_graph\ttarget_node\tdecoded_node_pairs\tdecoded_node\tlabel\n")
        proof_num_count = [0] * 10
        example_count = [0] * 10
        for (i, (record, meta_record)) in enumerate(zip(records, meta_records)):
            # print(i)
            if i > 5:
                break
            assert record["id"] == meta_record["id"]
            context = record["context"]
            # print(context)
            sentence_scramble = record["meta"]["sentenceScramble"]
            for (j, question) in enumerate(record["questions"]):
                # Uncomment to train/evaluate at a certain depth
                #if question["meta"]["QDep"] != 5:
                #    continue
                # Uncomment to test at a specific subset of Birds-Electricity dataset
                #if not record["id"].startswith("AttPosElectricityRB4"):
                #    continue

                id = question["id"]
                label = question["label"]
                question = question["text"]
                # print(question)
                meta_data = meta_record["questions"]["Q"+str(j+1)]
                question_depth = meta_data["QDep"]
                assert (question == meta_data["question"])

                proofs = meta_data["proofs"]
                example_count[question_depth] += 1
                proof_num = len(proofs.split("OR"))
                if proof_num > 1:
                    proof_num_count[question_depth] += 1

                nfact = meta_record["NFact"]
                nrule = meta_record["NRule"]
                node_pairs, node_label = self._get_node_labels(proofs, sentence_scramble, nfact, nrule)
                decoded_node_pairs = []

                decoded_node = [0] * (nfact + nrule + 2)
                for pair_index, node_pair in enumerate(node_pairs):
                    parent_node = node_pair[0]
                    child_node = node_pair[1]

                    target_graph = [0] * (nfact + nrule + 2)
                    target_node = [0] * (nfact + nrule + 2)
                    target_node[child_node] = 1

                    if pair_index == 0:
                        target_graph[-1] = 1
                    else:
                        parent_node_index = 0
                        for k in decoded_node[:parent_node]:
                            if k == 1:
                                parent_node_index += 1
                        target_graph[parent_node_index] = 1

                    output_file.write(str(id) + '\t' + context + '\t' + question + '\t' + str(target_graph)
                                      + '\t' + str(target_node) + '\t' + str(decoded_node_pairs)
                                      + '\t' + str(decoded_node) + '\t' + str(label) + '\n')
                    decoded_node_pairs.append(node_pair)
                    decoded_node[parent_node] = 1
                    decoded_node[child_node] = 1

        output_file.close()
        print(example_count)

def convert_examples_to_features_RR(examples,
                                 label_list,
                                 max_seq_length,
                                 max_node_length,
                                 max_edge_length,
                                 tokenizer,
                                 output_mode,
                                 cls_token_at_end=False,
                                 pad_on_left=False,
                                 cls_token='[CLS]',
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_token=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 cls_token_segment_id=1,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True):

    label_map = {label: i for i, label in enumerate(label_list)}
    # print(label_map)
    features = []
    max_size = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        sentences = sent_tokenize(example.context)
        context_tokens = []
        proof_offset = []
        for sentence in sentences:
            sentence_tokens = tokenizer.tokenize(sentence)
            context_tokens.extend(sentence_tokens)
            proof_offset.append(len(context_tokens))

        max_size = max(max_size, len(context_tokens))

        question_tokens = tokenizer.tokenize(example.question)
        proof_offset.append(proof_offset[-1]+len(question_tokens))

        special_tokens_count = 3 if sep_token_extra else 2
        _truncate_seq_pair(context_tokens, question_tokens, max_seq_length - special_tokens_count - 1)

        tokens = context_tokens + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        tokens += question_tokens + [sep_token]
        segment_ids += [sequence_b_segment_id] * (len(question_tokens) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        proof_offset = proof_offset + [0] * (max_node_length - len(proof_offset))
        # node_label = example.node_label
        # node_label = node_label + [-100] * (max_node_length - len(node_label))
        #
        # edge_label = example.edge_label
        # edge_label = edge_label + [-100] * (max_edge_length - len(edge_label))

        # attention_node = example.attention_node
        # attention_node = attention_node + [-100] * (max_node_length - len(attention_node))

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(proof_offset) == max_node_length
        # assert len(node_label) == max_node_length
        # assert len(edge_label) == max_edge_length

        label_id = label_map[example.label]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("id: %s" % (example.id))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("proof_offset: %s" % " ".join([str(x) for x in proof_offset]))
            # logger.info("node_label: %s" % " ".join([str(x) for x in node_label]))
            # logger.info("edge_label: %s" % " ".join([str(x) for x in edge_label]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            RRFeatures(id=id,
                       input_ids=input_ids,
                       input_mask=input_mask,
                       segment_ids=segment_ids,
                       proof_offset=proof_offset,
                       label_id=label_id))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        max_len = max(len(tokens_a), len(tokens_b), len(tokens_c))
        if max_len == len(tokens_a):
            tokens_a.pop()
        elif max_len == len(tokens_b):
            tokens_b.pop()
        else:
            tokens_c.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "rr" or task_name == "rr_qa":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

processors = {
    "rr": RRProcessor
}

output_modes = {
    "rr": "classification",
    "rr_qa": "classification"
}

GLUE_TASKS_NUM_LABELS = {
    "rr": 2,
    "rr_qa": 2
}


import torch
import pandas as pd
import ast

class InputExample(object):
    def __init__(self, id, context, question, target_graph, target_node, decoded_node_pairs, decoded_node, label):
        self.id = id
        self.context = context
        self.question = question
        self.target_graph = target_graph
        self.target_node = target_node
        self.decoded_node_pairs = decoded_node_pairs
        self.decoded_node = decoded_node
        self.label = label


class base_transform(object):
    def __init__(self, tokenizer, args, label_map):
        self.tokenizer = tokenizer
        self.args = args
        self.label_map = label_map

    def __call__(self, example):
        print(example.decoded_node_pairs)
        # label_map = {label: i for i, label in enumerate(label_list)}
        target_graph = example.target_graph
        target_node = example.target_node
        sentences = sent_tokenize(example.context)
        context_tokens = []
        proof_offset = []
        for sentence in sentences:
            sentence_tokens = self.tokenizer.tokenize(sentence)
            context_tokens.extend(sentence_tokens)
            proof_offset.append(len(context_tokens))

        question_tokens = self.tokenizer.tokenize(example.question)

        proof_offset.append(proof_offset[-1] + len(question_tokens)+1)
        special_tokens_count = 2
        _truncate_seq_pair(context_tokens, question_tokens, self.args.max_seq_length - special_tokens_count - 1)

        tokens = context_tokens + ['SEP']
        segment_ids = [0] * len(tokens)
        tokens += question_tokens + ['SEP']
        segment_ids += [1] * (len(question_tokens) + 1)
        tokens = ['CLS'] + tokens
        segment_ids = [1] + segment_ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = self.args.max_seq_length - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([0] * padding_length)
        proof_offset = proof_offset + [0] * (self.args.max_node_length - len(proof_offset))
        target_graph = target_graph + [0] * (self.args.max_node_length - len(target_graph))
        target_node = target_node + [0] * (self.args.max_node_length - len(target_node))
        decoded_node = example.decoded_node + [0] * (self.args.max_node_length - len(example.decoded_node))

        assert len(input_ids) == self.args.max_seq_length
        assert len(input_mask) == self.args.max_seq_length
        assert len(segment_ids) == self.args.max_seq_length
        assert len(proof_offset) == self.args.max_node_length

        label_id = self.label_map[example.label]
        input_ids = torch.tensor(input_ids)
        input_mask = torch.tensor(input_mask)
        segment_ids = torch.tensor(segment_ids)
        proof_offset = torch.tensor(proof_offset)
        target_graph = torch.tensor(target_graph)
        target_node = torch.tensor(target_node)
        decoded_node = torch.ByteTensor(decoded_node)
        label_id = torch.tensor(label_id)
        # self.decoded_node_pairs = decoded_node_pairs
        return {"id": example.id, "input_ids": input_ids, "segment_ids": segment_ids, "input_mask": input_mask,
                "proof_offset": proof_offset, "target_graph": target_graph, "target_node": target_node,
                "decoded_node": decoded_node, "label_id": label_id}

from torch.utils.data import Dataset
class Corpus(Dataset):
    def __init__(self, file_name, transform=None):
        super(Corpus, self).__init__()
        if os.path.isfile(file_name):
            self.df = pd.read_csv(os.path.join("./data/new_data", "train.csv"), delimiter="\t")
        self.transform = transform


    def __len__(self):
        if isinstance(self.df, pd.DataFrame):
            return self.df.shape[0]
        else:
            return 0

    def __getitem__(self, idx):
        data = self.df.iloc[idx, :]
        id = data['id']
        context = data['context']
        question = data['question']
        target_graph = ast.literal_eval(data['target_graph'])
        target_node = ast.literal_eval(data['target_node'])
        decoded_node_pairs = ast.literal_eval(data['decoded_node_pairs'])
        decoded_node = ast.literal_eval(data['decoded_node'])
        label = data['label']
        # print(decoded_node_pairs)

        example = InputExample(id, context, question, target_graph, target_node, decoded_node_pairs, decoded_node, label)
        example = self.transform(example)
        if example == 0:
            return 0
        return example

