import argparse
import os
import json
import sys
import pandas as pd
import ast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proof_utils import get_proof_graph, get_proof_graph_with_fail


def get_new_edges(nodes, edges):
    edges = list(reversed(edges))
    parent_nodes, levels_node = [], []
    if edges != []:
        parent_nodes.append(edges[0][1])
        while parent_nodes != []:
            parent_node = parent_nodes.pop(0)
            for x in edges:
                if x[1] == parent_node:
                    parent_nodes.append(x[0])
                    levels_node.append(x)

    return nodes, levels_node


def _get_node_labels(self, id, proofs, sentence_scramble, nfact, nrule):
    proof = proofs.split("OR")[0]
    if "FAIL" in proof:
        nodes, edges, str_dic = get_proof_graph_with_fail(proof, id)
    else:
        nodes, edges, str_dic = get_proof_graph(proof, id)
        nodes, edges = self.get_new_edges(nodes, edges)

    component_index_map = {}
    rule_facts_list = []
    for (i, index) in enumerate(sentence_scramble):
        if index <= nfact:
            component = "triple" + str(index)
            rule_facts_list.append(0)
        else:
            component = "rule" + str(index - nfact)
            rule_facts_list.append(1)
        component_index_map[component] = i
    component_index_map["NAF"] = len(sentence_scramble)
    rule_facts_list.append(2)

    new_parent_list, new_child_list, parent_nodes, node_pairs, node_list = [], [], [], [], []
    if edges != []:
        parent_nodes.append((edges[0][1], 0))
        node_list.append(edges[0][1])
        new_parent_list.append(0)
        new_child_list.append(1)
        for pair_index, node_pair in enumerate(edges):
            parent_node, parent_index = parent_nodes.pop(0)
            for x_index, x in enumerate(edges):
                if x[1] == parent_node:
                    parent_nodes.append((x[0], x_index + 1))
                    node_pairs.append([x_index + 1, parent_index])
                    node_list.append(x[0])
                    new_parent_list.append(parent_index + 1)
                    new_child_list.append(x_index + 1 + 1)
    new_node_list = []
    for node_index, node in enumerate(node_list):
        new_node_list.append(component_index_map[str_dic[node]])

    if len(nodes) == 1:
        new_node_list.append(component_index_map[str_dic[nodes[0]]])

    return node_pairs, new_node_list, component_index_map, nodes, edges, rule_facts_list, \
           new_parent_list, new_child_list


def get_node_labels(id, proofs, sentence_scramble, nfact, nrule):
    all_node_indices, all_node_pairs = [], []
    for proof in proofs.split("OR"):
        if "FAIL" in proof:
            nodes, edges, str_dic = get_proof_graph_with_fail(proof, id)
        else:
            nodes, edges, str_dic = get_proof_graph(proof, id)
            nodes, edges = get_new_edges(nodes, edges)
        component_index_map = {}
        for (i, index) in enumerate(sentence_scramble):
            if index <= nfact:
                component = "triple" + str(index)
            else:
                component = "rule" + str(index - nfact)
            component_index_map[component] = i
        component_index_map["NAF"] = len(sentence_scramble)

        parent_nodes, node_pairs, node_list = [], [], []
        if edges != []:
            parent_nodes.append((edges[0][1], 0))
            node_list.append(component_index_map[str_dic[edges[0][1]]])
            while parent_nodes != []:
                parent_node, parent_index = parent_nodes.pop(0)
                for x_index, x in enumerate(edges):
                    if x[1] == parent_node:
                        parent_nodes.append((x[0], x_index + 1))
                        node_pairs.append((component_index_map[str_dic[x[0]]], component_index_map[str_dic[x[1]]]))
                        node_list.append(component_index_map[str_dic[x[0]]])

        if len(nodes) == 1:
            node_list.append(component_index_map[str_dic[nodes[0]]])

        all_node_indices.append(node_list)
        all_node_pairs.append(node_pairs)
    return all_node_indices, all_node_pairs

strategy_label_map = {'proof': 1, 'inv-proof': 1, 'inv-rconc': 0, 'rconc': 0, 'random': 0, 'inv-random': 0}

def get_gold_proof_nodes_edges(data_dir, eval_split):
    test_file = os.path.join(data_dir, eval_split+".jsonl")
    meta_test_file = os.path.join(data_dir, "meta-"+eval_split+".jsonl")

    f1 = open(test_file, "r", encoding="utf-8-sig")
    f2 = open(meta_test_file, "r", encoding="utf-8-sig")

    gold_nodes, gold_edges = [], []
    gold_labels, gold_strategys = [], []
    ids = []
    proofs_list = []
    for (i, (record, meta_record)) in enumerate(zip(f1, f2)):
        # if i > 10:
        #     break
        record = json.loads(record)
        meta_record = json.loads(meta_record)

        sentence_scramble = record["meta"]["sentenceScramble"]
        for (j, question) in enumerate(record["questions"]):
            meta_data = meta_record["questions"]["Q" + str(j + 1)]
            id = question["id"]
            question_depth = meta_data["QDep"]
            proofs = meta_data["proofs"]
            nfact = meta_record["NFact"]
            nrule = meta_record["NRule"]
            label = question["label"]
            strategy = meta_data["strategy"]
            strategy_id = strategy_label_map[strategy]
            if question_depth not in [5, 4, 3, 2, 1, 0]:
                continue
            # if 'FAIL' in proofs:
            #     continue
            # if 'AttPosBirdsVar1' not in id:
            #     continue

            all_node_indices, all_node_pairs = get_node_labels(id, proofs, sentence_scramble, nfact, nrule)
            gold_nodes.append(all_node_indices)
            gold_edges.append(all_node_pairs)
            gold_labels.append(label)
            gold_strategys.append(strategy_id)
            ids.append(id)
            proofs_list.append(proofs)
    return gold_nodes, gold_edges, gold_labels, ids, proofs_list, gold_strategys

def get_result(args, eval_split=None):
    strategy_pred_file = args.output_dir + '/predictions_strategy_' + eval_split + '.csv'
    qa_pred_file = args.output_dir + '/predictions_' + eval_split + '.csv'
    node_pred_file = args.output_dir + '/prediction_nodes_' + eval_split + '.csv'
    edge_pred_file = args.output_dir + '/prediction_edge_' + eval_split + '.csv'
    node_pred_list_file = args.output_dir + '/prediction_nodes_list_' + eval_split + '.csv'
    edge_pred_list_file = args.output_dir + '/prediction_edge_list_' + eval_split + '.csv'

    if os.path.isfile(strategy_pred_file):
        strategy_pred = pd.read_csv(strategy_pred_file, delimiter="\t")

    if os.path.isfile(qa_pred_file):
        qa_pred = pd.read_csv(qa_pred_file, delimiter="\t")

    if os.path.isfile(node_pred_file):
        node_pred = pd.read_csv(node_pred_file, delimiter="\t")

    if os.path.isfile(edge_pred_file):
        edge_pred = pd.read_csv(edge_pred_file, delimiter="\t")

    if os.path.isfile(node_pred_list_file):
        node_pred_list = pd.read_csv(node_pred_list_file, delimiter="\t")

    if os.path.isfile(edge_pred_list_file):
        edge_pred_list = pd.read_csv(edge_pred_list_file, delimiter="\t")

    all_gold_nodes, all_gold_edges, all_gold_labels, ids, proofs_list, all_gold_strategys = get_gold_proof_nodes_edges(args.data_dir, eval_split)

    output_pred_file = os.path.join(args.output_dir, "truth_{}.csv".format(eval_split))
    with open(output_pred_file, "w") as writer:
        for index, qa_label in enumerate(all_gold_labels):
            writer.write(ids[index] + "\t")
            writer.write(str(qa_label) + "\n")

    output_strategy_pred_file = os.path.join(args.output_dir, "truth_strategy_{}.csv".format(eval_split))
    with open(output_strategy_pred_file, "w") as writer:
        for index, strategy_label in enumerate(all_gold_strategys):
            writer.write(ids[index] + "\t")
            writer.write(str(strategy_label) + "\n")

    output_node_pred_file = os.path.join(args.output_dir, "truth_nodes_{}.csv".format(eval_split))
    with open(output_node_pred_file, "w") as writer:
        for index, nodes in enumerate(all_gold_nodes):
            writer.write(ids[index] + "\t")
            writer.write(str(nodes) + "\n")

    output_edge_pred_file = os.path.join(args.output_dir, "truth_edge_{}.csv".format(eval_split))
    with open(output_edge_pred_file, "w") as writer:
        for index, edges in enumerate(all_gold_edges):
            writer.write(ids[index] + "\t")
            writer.write(str(edges) + "\n")

    all_pred_qa = qa_pred['qa_preds']
    all_pred_qa_id = qa_pred['id']
    all_pred_strategy = strategy_pred['strategy_preds']
    all_pred_strategy_id = strategy_pred['id']
    all_pred_nodes = node_pred['node_preds']
    all_pred_nodes_id = node_pred['id']
    all_pred_edges = edge_pred['edge_preds']
    all_pred_edges_id = edge_pred['id']
    all_sentence_lists = node_pred['sentence_lists']
    all_node_pred_list = node_pred_list['node_preds_list']
    all_edge_pred_list = edge_pred_list['edge_preds_list']

    assert len(all_gold_nodes) == len(all_pred_nodes)
    assert len(all_gold_edges) == len(all_pred_edges)
    assert len(all_node_pred_list) == len(all_pred_nodes)
    assert len(all_edge_pred_list) == len(all_pred_edges)
    assert len(all_gold_labels) == len(all_pred_qa)
    assert len(all_gold_labels) == len(all_pred_qa)

    print("Num samples = " + str(len(all_pred_qa)))

    correct_qa = 0
    correct_strategy = 0
    correct_nodes = 0
    correct_edges = 0
    correct_proofs = 0
    correct_samples = 0

    for i, qa_id in enumerate(all_pred_qa_id):
        is_correct_qa = False
        is_correct_strategys = False
        assert ids[i] == qa_id
        assert ids[i] == all_pred_nodes_id[i]
        assert ids[i] == all_pred_edges_id[i]
        assert ids[i] == all_pred_strategy_id[i]
        if all_gold_labels[i] == all_pred_qa[i]:
            is_correct_qa = True
            correct_qa += 1
        if all_gold_strategys[i] == all_pred_strategy[i]:
            is_correct_strategys = True
            correct_strategy += 1

        gold_nodes = all_gold_nodes[i]
        gold_edges = all_gold_edges[i]
        pred_node = ast.literal_eval(all_pred_nodes[i])
        pred_edge = ast.literal_eval(all_pred_edges[i])
        node_pred_list = ast.literal_eval(all_node_pred_list[i])
        edge_pred_list = ast.literal_eval(all_edge_pred_list[i])
        node_false_flag = True
        for (j, gold_node) in enumerate(gold_nodes):
            if set(gold_node) == set(pred_node):
                correct_nodes += 1
                node_false_flag = False
                break

        edge_false_flag = True
        for (j, gold_edge) in enumerate(gold_edges):
            if set(pred_edge) == set(gold_edge):
                correct_edges += 1
                edge_false_flag = False
                break

        is_correct_proof = False
        for (j, (gold_node, gold_edge)) in enumerate(zip(gold_nodes, gold_edges)):
            if set(gold_node) == set(pred_node) and set(pred_edge) == set(gold_edge):
                correct_proofs += 1
                is_correct_proof = True
                break

        if is_correct_proof and is_correct_qa:
            correct_samples += 1

    qa_accuracy = correct_qa / len(all_gold_labels)
    strategy_accuracy = correct_strategy / len(all_gold_strategys)
    node_accuracy = correct_nodes / len(all_gold_nodes)
    edge_accuracy = correct_edges / len(all_gold_edges)
    proof_accuracy = correct_proofs / len(all_gold_labels)
    full_accuracy = correct_samples / len(all_gold_labels)
    print('QA: ' + str(qa_accuracy))
    print('NA: ' + str(node_accuracy))
    print('EA: ' + str(edge_accuracy))
    print('PA: ' + str(proof_accuracy))
    print('FA: ' + str(full_accuracy))
    print('SA: ' + str(strategy_accuracy))
    return qa_accuracy, node_accuracy, edge_accuracy, proof_accuracy, full_accuracy
