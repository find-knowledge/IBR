import random
import torch
from torch import nn
import numpy as np

import csv
import logging
import os
import sys
from io import open

import json
from nltk.tokenize import sent_tokenize
import numpy as np

from proof_utils import get_proof_graph, get_proof_graph_with_fail
random.seed(42)


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
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train.jsonl")),
            self._read_jsonl(os.path.join(data_dir, "meta-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        # Change these to test paths for test results
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "dev.jsonl")),
            self._read_jsonl(os.path.join(data_dir, "meta-dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "test.jsonl")),
            self._read_jsonl(os.path.join(data_dir, "meta-test.jsonl")), "test")

    def get_labels(self):
        return [True, False]

    # Unconstrained training, use this for ablation
    def _get_node_labels(self, id, proofs, sentence_scramble, nfact, nrule):
        proof = proofs.split("OR")[0]
        if "FAIL" in proof:
            nodes, edges, str_dic = get_proof_graph_with_fail(proof, id)
            edges = list(reversed(edges))

        else:
             nodes, edges, str_dic = get_proof_graph(proof, id)

        component_index_map = {}
        rule_facts_list = []
        for (i, index) in enumerate(sentence_scramble):
            if index <= nfact:
                component = "triple" + str(index)
                rule_facts_list.append(0)
            else:
                component = "rule" + str(index-nfact)
                rule_facts_list.append(1)
            component_index_map[component] = i
        component_index_map["NAF"] = len(sentence_scramble)
        rule_facts_list.append(2)

        return component_index_map, nodes, edges, rule_facts_list, str_dic

    def _create_examples(self, records, meta_records, data_type):
        proof_num_count = [0] * 10
        example_count = [0] * 10
        examples = []
        strategy_dic = {}
        for (i, (record, meta_record)) in enumerate(zip(records, meta_records)):
            # if i > 10:
            #     break
            assert record["id"] == meta_record["id"]
            context = record["context"]
            # context = context.replace('If ', '')
            context = context.replace('is ', '')
            context = context.replace('are ', '')
            context = context.replace('do ', '')
            context = context.replace('does ', '')
            context = context.replace('the ', '')
            sentence_scramble = record["meta"]["sentenceScramble"]
            for (j, question) in enumerate(record["questions"]):
                id = question["id"]
                label = question["label"]
                question = question["text"]
                question = question.replace('is ', '')
                question = question.replace('are ', '')
                question = question.replace('do ', '')
                question = question.replace('does ', '')
                question = question.replace('the ', '')

                meta_data = meta_record["questions"]["Q"+str(j+1)]
                question_depth = meta_data["QDep"]
                strategy = meta_data["strategy"]
                if strategy not in strategy_dic.keys():
                    strategy_dic[strategy] = 0
                else:
                    strategy_dic[strategy] += 1
                # assert (question == meta_data["question"])

                proofs = meta_data["proofs"]
                example_count[question_depth] += 1
                proof_num = len(proofs.split("OR"))
                if proof_num > 1:
                    proof_num_count[question_depth] += 1

                nfact = meta_record["NFact"]
                nrule = meta_record["NRule"]

                component_index_map, nodes, edges, rule_facts_list, str_dic = self._get_node_labels(id, proofs, sentence_scramble, nfact, nrule)
                if question_depth in [5, 4, 3, 2, 1, 0]:

                    # if id != 'AttNoneg-D5-57-9':
                    #     continue
                    #
                    # print(id)
                    # print(question)
                    # print(proofs)
                    # print(123)
                    # if 'AttPosBirdsVar1' not in id:
                    #     continue

                    # if 'FAIL' in proofs:
                        # continue

                    examples.append(InputExample(id, context, question,
                                                 proofs.split("OR")[0], component_index_map, nodes,
                                                 edges, rule_facts_list,
                                                 str_dic, label, strategy))

        # print(strategy_dic)
        return examples


processors = {
    "rr": RRProcessor
}

processor = RRProcessor()
label_list = processor.get_labels()
label_map = {label: i for i, label in enumerate(label_list)}
strategy_label_map = {'proof': 1, 'inv-proof': 1, 'inv-rconc': 0, 'rconc': 0, 'random': 0, 'inv-random': 0}


def get_new_edges(nodes, edges, component_index_map, str_dic, rule_facts_list, id):
    edges = list(reversed(edges))
    parent_nodes, levels_node = [], []
    if edges != []:
        parent_nodes.append(edges[0][1])
        while parent_nodes != []:
            parent_node = parent_nodes.pop(0)
            fact_shuffle_list, rule_shuffle_list, naf_shuffle_list = [], [], []
            for x in edges:
                if x[1] == parent_node:
                    dic = {}
                    dic['parent_nodes'] = x[0]
                    dic['levels_node'] = x
                    if rule_facts_list[component_index_map[str_dic[x[0]]]] == 0:
                        fact_shuffle_list.append(dic)
                    elif rule_facts_list[component_index_map[str_dic[x[0]]]] == 1:
                        rule_shuffle_list.append(dic)
                    else:
                        naf_shuffle_list.append(dic)

            a = random.random()
            if a < 0.5:
                shuffle_list = naf_shuffle_list + fact_shuffle_list + rule_shuffle_list
            elif 0.5 <= a <= 1:
                shuffle_list = naf_shuffle_list + list(reversed(fact_shuffle_list)) + list(reversed(rule_shuffle_list))
            else:
                shuffle_list = naf_shuffle_list + rule_shuffle_list + fact_shuffle_list

            for k in shuffle_list:
                parent_nodes.append(k['parent_nodes'])
                levels_node.append(k['levels_node'])

    return nodes, levels_node

def ListsToTensor(xs, NIL):
    max_len = max(len(x) for x in xs)
    ys = []
    for i, x in enumerate(xs):
        y = x + [NIL] * (max_len - len(x))
        ys.append(y)
    data = np.transpose(np.array(ys))
    return data

id_change = {}

def batchify(data, args, tokenizer, sep_token_extra=True):
    examples = data
    concept, edge, input_ids, input_masks, segment_ids, proof_offsets, label_ids, strategy_ids \
        = [], [], [], [], [], [], [], []
    ids = []

    proofs, component_index_maps = [], []
    questions, contexts, questions_contexts, node_length = [], [], [], []
    rule_facts_lists, new_parent_lists, new_child_lists, sentence_lists, brother_lists = [], [], [], [], []

    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    DUM = args.max_node_length
    END = args.max_node_length + 1
    NIL = args.max_node_length + 2

    max_input_id_length = 0
    for example_index, example in enumerate(examples):
        question_tokens = tokenizer.tokenize(example.question)
        sentences = sent_tokenize(example.context)
        context_tokens = []
        for sentence_index, sentence in enumerate(sentences):
            sentence_tokens = tokenizer.tokenize(sentence)
            context_tokens.extend(sentence_tokens)
        tokens = question_tokens + [sep_token]
        if sep_token_extra:
            tokens += [sep_token]
        tokens += context_tokens + [sep_token]

        tokens = [cls_token] + tokens
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        if len(input_id) > max_input_id_length:
            max_input_id_length = len(input_id)
    max_sentence_length_list, sentence_count_list = [], []
    for example_index, example in enumerate(examples):
        max_sentence_length = 0
        sentence_count = 0
        if example.id not in id_change.keys():
            id_change[example.id] = 0
        else:
            id_change[example.id] += 1

        edges = example.edges
        nodes = example.nodes
        component_index_map = example.component_index_map
        str_dic = example.str_dic
        nodes, edges = get_new_edges(nodes, edges, component_index_map, str_dic, example.rule_facts_list, example.id)
        new_parent_list, new_child_list, parent_nodes, node_pairs, node_list, brother_list = [], [], [], [], [], []
        if edges != []:
            parent_nodes.append((edges[0][1], 0))
            node_list.append(edges[0][1])
            new_parent_list.append(0)
            new_child_list.append(1)
            brother_list.append([])
            while parent_nodes != []:
                parent_node, parent_index = parent_nodes.pop(0)
                brother_count = 0
                temp_brother = []
                for x_index, x in enumerate(edges):
                    if x[1] == parent_node:
                        parent_nodes.append((x[0], x_index + 1))
                        node_pairs.append([x_index + 1, parent_index])
                        node_list.append(x[0])
                        new_parent_list.append(parent_index + 1)
                        new_child_list.append(x_index + 1 + 1)
                        brother_count += 1
                        brother_list.append(temp_brother.copy())
                        temp_brother.append(x_index + 1 + 1)

            new_parent_list.append(len(new_parent_list))
        new_node_list = []
        for node_index, node in enumerate(node_list):
            new_node_list.append(component_index_map[str_dic[node]])

        if len(nodes) == 1:
            new_node_list.append(component_index_map[str_dic[nodes[0]]])
            new_parent_list.append(0)
            # new_parent_list.append(-1)
            new_parent_list.append(len(new_parent_list))
            brother_list.append([])
        if len(nodes) == 0:
            new_parent_list.append(0)

        proofs.append(example.proof)
        component_index_maps.append(component_index_map)

        rule_facts_lists.append(example.rule_facts_list)
        new_parent_lists.append(new_parent_list)
        new_child_lists.append(new_child_list)
        brother_lists.append(brother_list)

        questions.append(example.question)
        contexts.append(example.context)
        ids.append(example.id)
        concept.append(new_node_list)

        edge.append(node_pairs)
        sentences = sent_tokenize(example.context)
        context_tokens = []
        proof_offset = []
        question_tokens = tokenizer.tokenize(example.question)
        if len(question_tokens) > max_sentence_length:
            max_sentence_length = len(question_tokens)
        sentence_count += 1
        proof_offset.append(len(question_tokens)+1)
        sentence_list = []
        for sentence in sentences:
            sentence_list.append(sentence)
            sentence_tokens = tokenizer.tokenize(sentence)
            context_tokens.extend(sentence_tokens)
            if len(sentence_tokens) > max_sentence_length:
                max_sentence_length = len(sentence_tokens)
            sentence_count += 1
            proof_offset.append(len(context_tokens)+len(question_tokens)+1)
        max_sentence_length_list.append(max_sentence_length)
        sentence_count_list.append(sentence_count)
        special_tokens_count = 3 if sep_token_extra else 2
        _truncate_seq_pair(question_tokens, context_tokens, max_input_id_length - special_tokens_count - 1)

        tokens = question_tokens + [sep_token]
        if sep_token_extra:
            tokens += [sep_token]
        segment_id = [0] * len(tokens)
        tokens += context_tokens + [sep_token]
        segment_id += [1] * (len(context_tokens) + 1)

        tokens = [cls_token] + tokens
        segment_id = [0] + segment_id
        questions_contexts.append(tokens)
        input_id = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_id)
        # Zero-pad up to the sequence length.
        padding_length = max_input_id_length - len(input_id)
        input_id = input_id + ([pad_token] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_id = segment_id + ([0] * padding_length)
        node_length.append(len(proof_offset))
        proof_offset = proof_offset + [0] * (args.max_node_length - len(proof_offset))
        label_id = label_map[example.label]
        strategy_id = strategy_label_map[example.strategy]
        assert len(input_id) == max_input_id_length
        assert len(input_mask) == max_input_id_length
        assert len(segment_id) == max_input_id_length
        assert len(proof_offset) == args.max_node_length
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        proof_offsets.append(proof_offset)
        label_ids.append(label_id)
        strategy_ids.append(strategy_id)
        sentence_lists.append(sentence_list)

    input_ids = torch.tensor(input_ids)
    input_masks = torch.tensor(input_masks)
    segment_ids = torch.tensor(segment_ids)
    proof_offsets = torch.tensor(proof_offsets)
    label_ids = torch.tensor(label_ids)
    strategy_ids = torch.tensor(strategy_ids)

    augmented_concept = [[DUM] + x + [END] for x in concept]
    augmented_concept_in = [[DUM] + x for x in concept]
    _concept_in = ListsToTensor(augmented_concept_in, NIL)
    _concept_out = ListsToTensor(augmented_concept, NIL)[1:]
    out_conc_len, bsz = _concept_out.shape
    _rel = np.full((out_conc_len, bsz, out_conc_len), 0)
    _conc = np.full((out_conc_len, bsz, args.max_node_length+3), 0)
    for bidx, (x, y) in enumerate(zip(edge, concept)):
        for l in range(1, len(y)+1):
            _rel[l, bidx, 1:l + 1] = 2
        for v, u in x:
            _rel[v, bidx, u+1] = 1  # v: [concept_1, ..., concept_n, <end>] u: [<dummy>, concept_1, ..., concept_n}]
        if x:
            _rel[v+1, bidx, len(y)] = 1
        else:
            if y:
                _rel[1, bidx, len(y)] = 1

    for bidx, y in enumerate(node_length):
        _conc[:, bidx, y:] = 2
    _conc[:, :, END] = 0
    for bidx, x in enumerate(np.transpose(np.array(_concept_out))):
        for index, u in enumerate(x):
            _conc[index, bidx, u] = 1
            if u == NIL:
                _conc[index, bidx, :] = 2

    _concept_in = torch.from_numpy(_concept_in)
    _concept_out = torch.from_numpy(_concept_out)
    _rel = torch.from_numpy(_rel)
    _conc = torch.from_numpy(_conc)
    return {"ids": ids, "input_ids": input_ids, "segment_ids": segment_ids, "input_mask": input_masks,
            "proof_offset": proof_offsets, 'concept_in': _concept_in, 'concept_out': _concept_out, 'rel': _rel,
            "label_id": label_ids, 'questions_contexts': questions_contexts, 'proofs': proofs, 'component_index_maps': component_index_maps,
            'conc': _conc, 'rule_facts_lists': rule_facts_lists, 'brother_lists': brother_lists, 'max_sentence_length_list': max_sentence_length_list,
            'new_parent_lists': new_parent_lists, 'new_child_lists': new_child_lists, 'sentence_lists': sentence_lists,
            'sentence_count_list': sentence_count_list, 'strategy_id': strategy_ids}



class DataLoader(object):
    def __init__(self, data, batch_size, batch_size_fix, args, tokenizer, epoch_seed, for_train):
        self.data = data
        self.batch_size = batch_size
        self.batch_size_fix = batch_size_fix
        self.train = for_train
        self.epoch_seed = epoch_seed
        self.args = args
        self.tokenizer = tokenizer
        if self.train:
            self.GPU_SIZE = args.GPU_SIZE
        else:
            self.GPU_SIZE = args.GPU_SIZE_eval

    def __len__(self):
        r = random.random
        random.seed(42)
        idx = list(range(len(self.data)))
        if self.train:
            random.shuffle(idx, random=r)
            # print(idx)
            # idx.sort(key=lambda x: len(self.data[x].context) + len(self.data[x].question), reverse=True)
        batches = []
        num_tokens, data = 0, []
        for i in idx:
            num_tokens += len(self.data[i].context) + len(self.data[i].question)
            data.append(self.data[i])
            if num_tokens >= self.batch_size:
                sz = len(data) * (2 + max(len(x.context) for x in data) + max(len(x.question) for x in data))
                if sz > self.GPU_SIZE:
                    # because we only have limited GPU memory
                    batches.append(data[:len(data) // 2])
                    data = data[len(data) // 2:]
                batches.append(data)
                num_tokens, data = 0, []
            if len(data) == self.batch_size_fix:
                batches.append(data)
                num_tokens, data = 0, []

        if data:
            batches.append(data)
        # print(len(batches))
        return len(batches)

    def __iter__(self):
        r = random.random
        random.seed(42)
        idx = list(range(len(self.data)))

        if self.train:
            random.shuffle(idx, random=r)
            # print(idx)
            # idx.sort(key=lambda x: len(self.data[x].context) + len(self.data[x].question), reverse=True)

        batches = []
        num_tokens, data = 0, []
        for i in idx:
            num_tokens += len(self.data[i].context) + len(self.data[i].question)
            data.append(self.data[i])
            if num_tokens >= self.batch_size:
                sz = len(data) * (2 + max(len(x.context) for x in data) + max(len(x.question) for x in data))
                if sz > self.GPU_SIZE:
                    # because we only have limited GPU memory
                    batches.append(data[:len(data) // 2])
                    data = data[len(data) // 2:]
                batches.append(data)
                num_tokens, data = 0, []
            if len(data) == self.batch_size_fix:
                batches.append(data)
                num_tokens, data = 0, []

        if data:
            batches.append(data)

        # if self.train:
        #     random.shuffle(batches)
        # print(len(batches))
        # r = random.random
        random.seed(self.epoch_seed)

        for batch in batches:
            yield batchify(batch, self.args, self.tokenizer)

class InputExample(object):
    def __init__(self, id, context, question, proof,
                 component_index_map, nodes, edges, rule_facts_list,
                 str_dic, label, strategy):
        self.id = id
        self.context = context
        self.question = question
        self.proof = proof
        self.component_index_map = component_index_map
        self.nodes = nodes
        self.edges = edges
        self.rule_facts_list = rule_facts_list
        self.str_dic = str_dic
        self.label = label
        self.strategy = strategy


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "rr" or task_name == "rr_qa":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)