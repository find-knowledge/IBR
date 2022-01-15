import torch
import numpy as np
import argparse
import logging
import os
import random
from itertools import cycle
from tqdm import tqdm, trange
from pytorch_transformers import WarmupLinearSchedule, BertConfig, BertTokenizer, RobertaConfig, RobertaTokenizer, \
    WarmupCosineSchedule
from transformers import get_polynomial_decay_schedule_with_warmup
from torch.optim import AdamW
from utils import output_modes
from data_new import processors
from model.model import BertForPRover
from data_new import DataLoader
from eval_proof import get_result


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

MODEL_CLASSES = {
    'bert': (RobertaConfig, BertForPRover, RobertaTokenizer),
    # 'roberta': (RobertaConfig, RobertaForPRover, RobertaTokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def load_model(Model, args, config):
    if os.path.isfile(args.resume_model_path) and args.to_resume_model:
        model = Model(config=config)
        logger.info("resuming model from {} ...".format(args.resume_model_path))
        model.load_state_dict(torch.load(args.resume_model_path))
    else:
        model = Model.from_pretrained(args.model_name_or_path, config=config)
    return model

def load_and_cache_examples(args, task, eval_split):
    processor = processors[task]()
    if eval_split == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif eval_split == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    elif eval_split == "test":
        examples = processor.get_test_examples(args.data_dir)
    return examples

def train(args):
    """ Train the model """
    args.output_mode = output_modes[args.task_name]
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path, num_labels=2, finetuning_task=args.task_name)
    config = add_args_to_config(args, config)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    model.to(args.device)
    examples = load_and_cache_examples(args, args.task_name, 'train')
    train_dataloader = DataLoader(examples, args.batch_size, args.train_batch_size, args, tokenizer, 42, for_train=True)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs

    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer]
    bert_no_decay = [n for n, p in param_optimizer if 'bert' in n and ('bias' in n or 'LayerNorm.weight' in n or 'layer_norm' in n)]
    bert_decay = [n for n, p in param_optimizer if 'bert' in n and not ('bias' in n or 'LayerNorm.weight' in n or 'layer_norm' in n)]
    beside_bert_no_decay = [n for n, p in param_optimizer if 'bert' not in n and ('bias' in n or 'LayerNorm.weight' in n or 'layer_norm' in n)]
    beside_bert_decay = [n for n, p in param_optimizer if 'bert' not in n and not ('bias' in n or 'LayerNorm.weight' in n or 'layer_norm' in n)]
    arc_generator_no_decay = [n for n, p in param_optimizer if 'arc' in n and ('bias' in n or 'LayerNorm.weight' in n or 'layer_norm' in n)]
    arc_generator_decay = [n for n, p in param_optimizer if 'arc' in n and not ('bias' in n or 'LayerNorm.weight' in n or 'layer_norm' in n)]
    all_embeddings = [n for n, p in param_optimizer if 'bert' not in n and 'embeddings' in n]
    lstm_no_decay = [n for n, p in param_optimizer if 'lstm' in n and ('bias' in n or 'LayerNorm.weight' in n or 'layer_norm' in n)]
    lstm_decay = [n for n, p in param_optimizer if 'lstm' in n and not ('bias' in n or 'LayerNorm.weight' in n or 'layer_norm' in n)] + all_embeddings
    beside_bert_no_decay = list(set(beside_bert_no_decay) - set(lstm_no_decay) - set(arc_generator_no_decay))
    beside_bert_decay = list(set(beside_bert_decay) - set(lstm_decay) - set(arc_generator_decay))
    optimizer_grouped_parameters_bert = [
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in bert_decay)],
         'lr': args.bert_learning_rate, 'weight_decay': args.bert_weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in bert_no_decay)],
         'lr': args.bert_learning_rate, 'weight_decay': 0.}]
    optimizer_grouped_parameters_beside_bert = [
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in beside_bert_decay)],
         'weight_decay': args.beside_bert_weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in beside_bert_no_decay)],
         'weight_decay': 0.}]
    optimizer_grouped_parameters_lstm = [
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in lstm_decay)],
         'weight_decay': args.beside_bert_weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in lstm_no_decay)],
         'weight_decay': 0.}]
    optimizer_grouped_parameters_arc_generator = [
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in arc_generator_decay)],
         'weight_decay': args.arc_generator_weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in arc_generator_no_decay)],
         'weight_decay': 0.}]
    optimizer_bert = AdamW(optimizer_grouped_parameters_bert, eps=args.adam_epsilon)
    optimizer_beside_bert = AdamW(optimizer_grouped_parameters_beside_bert,
                                  lr=args.beside_bert_learning_rate, eps=args.adam_epsilon)
    optimizer_lstm = AdamW(optimizer_grouped_parameters_lstm,
                           lr=args.lstm_learning_rate, eps=args.adam_epsilon)
    optimizer_arc_generator = AdamW(optimizer_grouped_parameters_arc_generator,
                                    lr=args.arc_generator_rate, eps=args.adam_epsilon)

    scheduler_bert = WarmupLinearSchedule(optimizer_bert, warmup_steps=args.bert_warmup_steps, t_total=t_total)
    scheduler_beside_bert = get_polynomial_decay_schedule_with_warmup(
        optimizer_beside_bert, num_warmup_steps=args.beside_bert_warmup_steps, num_training_steps=t_total, power=4.0)
    scheduler_arc_generator = get_polynomial_decay_schedule_with_warmup(
        optimizer_arc_generator, num_warmup_steps=args.beside_bert_warmup_steps, num_training_steps=t_total, power=4.0)
    scheduler_lstm = WarmupLinearSchedule(optimizer_lstm, warmup_steps=args.beside_bert_warmup_steps, t_total=t_total)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Train!
    logger.info("***** Running training *****")
    logger.info("Num Epochs = %d", args.epochs)
    logger.info("batch step = %d", len(train_dataloader))
    logger.info("Total optimization steps = %d", t_total)
    global_step, total_qa_loss, total_arc_loss, total_concept_loss, total_strategy_loss = 0, 0, 0, 0, 0
    # model.zero_grad()
    batches_acm = 0
    print_step = 0
    step = 0
    for epoch_count in range(args.epochs):
        train_dataloader = DataLoader(examples, args.batch_size, args.train_batch_size, args, tokenizer, epoch_count, for_train=True)
        bar = tqdm(range(len(train_dataloader)), total=len(train_dataloader))
        train_loader = cycle(train_dataloader)

        for epoch_step in bar:
            # print(step)
            model.train()
            data_batch = next(train_loader)
            for k, v in data_batch.items():
                if k not in ['ids', 'questions', 'contexts', 'questions_contexts', 'proofs', 'sentence_count_list',
                             'component_index_maps', 'rule_facts_lists', 'brother_lists', 'max_sentence_length_list',
                             'new_parent_lists', 'new_child_lists', 'sentence_lists']:
                    data_batch[k] = v.to(args.device)

            qa_loss, arc_loss, concept_loss, strategy_loss = model(batch=data_batch, step=step / (len(train_dataloader) * args.epochs))
            loss = qa_loss + arc_loss + concept_loss + 0.3*strategy_loss
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            print_step += 1
            total_qa_loss += qa_loss.item()
            total_arc_loss += arc_loss.item()
            total_concept_loss += concept_loss.item()
            total_strategy_loss += strategy_loss.item()
            bar.set_description("loss {}".format(loss.item() * args.gradient_accumulation_steps))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                batches_acm += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer_bert.step()
                scheduler_bert.step()  # Update learning rate schedule
                optimizer_bert.zero_grad()

                optimizer_beside_bert.step()
                scheduler_beside_bert.step()  # Update learning rate schedule
                optimizer_beside_bert.zero_grad()

                optimizer_arc_generator.step()
                scheduler_arc_generator.step()  # Update learning rate schedule
                optimizer_arc_generator.zero_grad()

                optimizer_lstm.step()
                scheduler_lstm.step()  # Update learning rate schedule
                optimizer_lstm.zero_grad()

                global_step += 1
                if batches_acm % args.print_steps == 0:
                    # print(optimizer_bert.state_dict()['param_groups'][0]['lr'])
                    # print(optimizer_bert.state_dict()['param_groups'][1]['lr'])
                    # print(optimizer_beside_bert.state_dict()['param_groups'][0]['lr'])
                    # print(optimizer_beside_bert.state_dict()['param_groups'][1]['lr'])
                    logger.info("=========train report =========")
                    logger.info("step : %s ", str(global_step))
                    logger.info("average_qa loss: %s" % (str(total_qa_loss / print_step)))
                    logger.info("average_arc loss: %s" % (str(total_arc_loss / print_step)))
                    logger.info("average_concept loss: %s" % (str(total_concept_loss / print_step)))
                    logger.info("average_strategy loss: %s" % (str(total_strategy_loss / print_step)))

                    # output_eval_file = os.path.join(args.output_dir, "train_records.txt")
                    # with open(output_eval_file, "a+") as writer:
                    #     writer.write("=========train report =========\n")
                    #     writer.write("step : %s \n" % (str(global_step)))
                    #     writer.write("average_qa loss: %s\n" % (str(total_qa_loss / print_step)))
                    #     writer.write("average_arc loss: %s\n" % (str(total_arc_loss / print_step)))
                    #     writer.write("average_concept loss: %s\n" % (str(total_concept_loss / print_step)))
                    #     writer.write("average_strategy loss: %s\n" % (str(total_strategy_loss / print_step)))
                    #     writer.write('\n')

                    total_qa_loss = 0
                    total_arc_loss = 0
                    total_concept_loss = 0
                    total_strategy_loss = 0
                    print_step = 0

                if batches_acm % args.eval_steps == 0:
                    qa_accuracy, node_accuracy, edge_accuracy, proof_accuracy, full_accuracy = \
                        evaluate(args, model=model, tokenizer=tokenizer, eval_split="test", work=False)
                    logger.info("=========test report =========")
                    logger.info("step : %s ", str(global_step))
                    logger.info("qa_accuracy : %s" % (str(qa_accuracy)))
                    logger.info("node_accuracy : %s" % (str(node_accuracy)))
                    logger.info("edge_accuracy : %s" % (str(edge_accuracy)))
                    logger.info("proof_accuracy : %s" % (str(proof_accuracy)))
                    logger.info("full_accuracy : %s" % (str(full_accuracy)))

                    output_eval_file = os.path.join(args.output_dir, "test_records.txt")
                    with open(output_eval_file, "a+") as writer:
                        writer.write("=========test report =========\n")
                        writer.write("step : %s \n" % (str(global_step)))
                        writer.write("qa_accuracy : %s\n" % (str(qa_accuracy)))
                        writer.write("node_accuracy : %s\n" % (str(node_accuracy)))
                        writer.write("edge_accuracy : %s\n" % (str(edge_accuracy)))
                        writer.write("proof_accuracy : %s\n" % (str(proof_accuracy)))
                        writer.write("full_accuracy : %s\n" % (str(full_accuracy)))
                        writer.write('\n')

                    output_path = os.path.join(args.output_dir, "pytorch_model.bin")
                    if hasattr(model, 'module'):
                        logger.info("model has module")
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), output_path)
                    logger.info("model saved")

            step += 1


def parse_batch(model, batch, beam_size, alpha, max_time_step):
    res = dict()
    concept_batch = []
    edge_batch = []
    beams, logits, strategy_logits = model.work(batch, beam_size, max_time_step)
    score_batch = []
    ids = batch['ids']
    concept_list_batch, edge_list_batch, score_list_batch = [], [], []
    for beam_index, beam in enumerate(beams):
        predicted_concept_list, predicted_rel_list, score_list = [], [], []
        for hyp in beam.get_k_best(4, alpha):
            best_hyp = hyp
            predicted_concept = [token for token in best_hyp.seq[1:-1]]
            predicted_rel, predicted_rel_2 = [], []
            for i in range(len(predicted_concept)):
                if i == 0:
                    continue
                arc = best_hyp.state_dict['arc_ll%d' % i].squeeze_().exp_()[1:]  # head_len
                arc_max, arc_max_index = 0, 0
                for head_id, arc_prob in enumerate(arc.tolist()):
                    predicted_rel_2.append((predicted_concept[i], predicted_concept[head_id], arc_prob))
                    if arc_prob >= arc_max:
                        arc_max = arc_prob
                        arc_max_index = head_id

                predicted_rel.append((predicted_concept[i], predicted_concept[arc_max_index]))

            predicted_concept_list.append(predicted_concept)
            predicted_rel_list.append(predicted_rel)
            score_list.append(best_hyp.score)

        concept_batch.append(predicted_concept_list[0])
        score_batch.append(score_list[0])
        edge_batch.append(predicted_rel_list[0])
        concept_list_batch.append(predicted_concept_list[1:])
        score_list_batch.append(score_list[1:])
        edge_list_batch.append(predicted_rel_list[1:])

    res['concept'] = concept_batch
    res['score'] = score_batch
    res['edge'] = edge_batch
    res['concept_list'] = concept_list_batch
    res['score_list'] = score_list_batch
    res['edge_list'] = edge_list_batch
    return res, logits, strategy_logits


def evaluate(args, model=None, tokenizer=None, eval_split=None, work=False):
    if (work and eval_split == 'test') or (work and eval_split == 'train'):
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(args.model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        config = add_args_to_config(args, config)
        model = model_class.from_pretrained(args.evaluate_model_name_or_path, config=config)
        model.to(args.device)

    examples = load_and_cache_examples(args, args.task_name, eval_split)
    dataloader = DataLoader(examples, args.batch_size_eval, args.eval_batch_size, args, tokenizer, 42, for_train=False)

    processor = processors[args.task_name]()
    eval_output_dir = args.output_dir
    steps = tqdm(list(range(len(dataloader))), total=len(dataloader))
    dataloader = cycle(dataloader)

    qa_preds, node_preds, edge_preds, id_preds, score_preds, sentence_lists = [], [], [], [], [], []
    node_preds_lists, edge_preds_lists, score_preds_lists = [], [], []
    strategy_preds = []
    for step in steps:
        model.eval()
        data_batch = next(dataloader)
        for k, v in data_batch.items():
            if k not in ['ids', 'questions', 'contexts', 'questions_contexts', 'proofs', 'sentence_count_list',
                         'component_index_maps', 'rule_facts_lists', 'brother_lists', 'max_sentence_length_list',
                         'new_parent_lists', 'new_child_lists', 'sentence_lists']:
                data_batch[k] = v.to(args.device)
        with torch.no_grad():
            res, logits, strategy_logits = parse_batch(model, data_batch, args.beam_size, args.alpha, args.max_time_step)

        qa_preds.append(logits.detach().cpu().numpy())
        strategy_preds.append(strategy_logits.detach().cpu().numpy())
        node_preds.append(res['concept'])
        edge_preds.append(res['edge'])
        score_preds.append(res['score'])
        node_preds_lists.append(res['concept_list'])
        edge_preds_lists.append(res['edge_list'])
        score_preds_lists.append(res['score_list'])
        id_preds.append(data_batch['ids'])
        sentence_lists.append(data_batch['sentence_lists'])

    # The model outputs the QA accuracy, QA predictions, node predictions and the edge logit predictions

    # QA Predictions
    output_pred_file = os.path.join(eval_output_dir, "predictions_{}.csv".format(eval_split))

    with open(output_pred_file, "w") as writer:
        logger.info("***** Write predictions qa on {} *****".format(eval_split))
        writer.write("id" + "\t" + "qa_preds" + "\n")
        for batch_index in range(len(qa_preds)):
            for index, qa_pred in enumerate(qa_preds[batch_index]):
                writer.write(id_preds[batch_index][index] + "\t")
                writer.write(str(processor.get_labels()[qa_pred]) + "\n")

    # strategy Predictions
    output_strategy_file = os.path.join(eval_output_dir, "predictions_strategy_{}.csv".format(eval_split))

    with open(output_strategy_file, "w") as writer:
        logger.info("***** Write predictions strategy on {} *****".format(eval_split))
        writer.write("id" + "\t" + "strategy_preds" + "\n")
        for batch_index in range(len(strategy_preds)):
            for index, strategy_pred in enumerate(strategy_preds[batch_index]):
                writer.write(id_preds[batch_index][index] + "\t")
                writer.write(str(strategy_pred) + "\n")

    # prediction nodes
    output_node_pred_file = os.path.join(eval_output_dir, "prediction_nodes_{}.csv".format(eval_split))
    with open(output_node_pred_file, "w") as writer:
        logger.info("***** Write predictions nodes on {} *****".format(eval_split))
        writer.write("id" + "\t" + "node_preds" + "\t" + "edge_preds" + "\t" + "sentence_lists" + "\n")
        for batch_index in range(len(node_preds)):
            for index, node_pred in enumerate(node_preds[batch_index]):
                writer.write(id_preds[batch_index][index] + "\t")
                writer.write(str(list(node_pred)) + "\t")
                writer.write(str(list(edge_preds[batch_index][index])) + "\t")
                writer.write(str(sentence_lists[batch_index][index]) + "\n")

    output_node_pred_file = os.path.join(eval_output_dir, "prediction_nodes_list_{}.csv".format(eval_split))
    with open(output_node_pred_file, "w") as writer:
        logger.info("***** Write predictions nodes_list on {} *****".format(eval_split))
        writer.write("id" + "\t" + "node_preds_list" + "\n")
        for batch_index in range(len(node_preds_lists)):
            for index, node_preds_list in enumerate(node_preds_lists[batch_index]):
                writer.write(id_preds[batch_index][index] + "\t")
                writer.write(str(node_preds_list) + "\n")

    # prediction edge logits
    output_edge_pred_file = os.path.join(eval_output_dir, "prediction_edge_{}.csv".format(eval_split))
    with open(output_edge_pred_file, "w") as writer:
        logger.info("***** Write predictions edges on {} *****".format(eval_split))
        writer.write("id" + "\t" + "edge_preds" + "\n")
        for batch_index in range(len(edge_preds)):
            for index, edge_pred in enumerate(edge_preds[batch_index]):
                writer.write(id_preds[batch_index][index] + "\t")
                writer.write(str(list(edge_pred)) + "\n")

    output_edge_pred_file = os.path.join(eval_output_dir, "prediction_edge_list_{}.csv".format(eval_split))
    with open(output_edge_pred_file, "w") as writer:
        logger.info("***** Write predictions edges_list on {} *****".format(eval_split))
        writer.write("id" + "\t" + "edge_preds_list" + "\n")
        for batch_index in range(len(edge_preds_lists)):
            for index, edge_preds_list in enumerate(edge_preds_lists[batch_index]):
                writer.write(id_preds[batch_index][index] + "\t")
                writer.write(str(edge_preds_list) + "\n")

    return get_result(args, eval_split)


def add_args_to_config(args, config):
    config.DUM = args.max_node_length
    config.END = args.max_node_length + 1
    config.NIL = args.max_node_length + 2
    config.num_labels = 2
    config.max_node_length = args.max_node_length
    config.output_dir = args.output_dir
    config.position_size = args.max_node_length + 3
    config.device = args.device

    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo", default='running IBR model ', type=str, required=False)
    parser.add_argument("--data_dir", default='./data/depth-5', type=str, required=False)
    parser.add_argument("--model_type", default='bert', type=str, required=False)
    parser.add_argument("--model_name_or_path", default='./roberta-large', type=str, required=False)
    parser.add_argument("--evaluate_model_name_or_path", default='./output/d5/pytorch_model.bin', type=str, required=False)

    parser.add_argument("--task_name", default='rr', type=str, required=False)
    parser.add_argument("--output_dir", default='./output/d5', type=str, required=False)
    parser.add_argument("--data_cache_dir", default='./output/cache/', type=str, required=False)
    parser.add_argument("--do_train", action='store_true', default=True, required=False)
    parser.add_argument("--do_eval", action='store_true', default=False, required=False)
    parser.add_argument("--do_prediction", action='store_true', default=False, required=False)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--bert_learning_rate", default=1e-5, type=float)
    parser.add_argument("--beside_bert_learning_rate", default=5e-4, type=float)
    parser.add_argument("--arc_generator_rate", default=2e-4, type=float)
    parser.add_argument("--lstm_learning_rate", default=1e-3, type=float)
    parser.add_argument("--beside_bert_weight_decay", default=5e-4, type=float)
    parser.add_argument("--arc_generator_weight_decay", default=1e-3, type=float)
    parser.add_argument("--bert_weight_decay", default=5e-2, type=float)
    parser.add_argument("--epochs", default=8, type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--batch_size", default=100000, type=int)
    parser.add_argument("--GPU_SIZE", default=120000, type=int)
    parser.add_argument("--batch_size_eval", default=100000, type=int)
    parser.add_argument("--GPU_SIZE_eval", default=120000, type=int)
    parser.add_argument("--eval_steps", default=4356, type=int, required=False)
    parser.add_argument("--print_steps", default=1000, type=int, required=False)
    parser.add_argument("--train_eval_steps", default=9000000, type=int, required=False)
    parser.add_argument('--beam_size', default=8, type=int)
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--max_time_step', default=26, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    ## Other parameters
    parser.add_argument("--max_seq_length", default=300, type=int)
    parser.add_argument("--max_edge_length", default=676, type=int)
    parser.add_argument("--max_node_length", default=26, type=int)
    parser.add_argument("--do_lower_case", default=True, action='store_true')
    parser.add_argument("--bert_warmup_steps", default=0, type=int)
    parser.add_argument("--beside_bert_warmup_steps", default=3000, type=int)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--warmup_pct", default=None, type=float,
                        help="Linear warmup over warmup_pct*total_steps.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)

    # get_result(args, "train")
    # Training
    if args.do_train:
        train(args)

    if args.do_eval:
        logger.info("Prediction on the dev set")
        evaluate(args, eval_split="dev", work=True)

    if args.do_prediction:
        logger.info("Prediction on the test set")
        # get_result(args, "test")
        evaluate(args, eval_split="test", work=True)

    logger.info("***** Experiment finished *****")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    main()
