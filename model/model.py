from pytorch_transformers import BertPreTrainedModel, RobertaConfig, \
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, RobertaModel, BertModel
from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn
from model.transformer import MultiheadAttention, Transformer, SelfAttentionMask
from torch.nn import functional as F
from search import Hypothesis, Beam, search_by_batch
from data_new import ListsToTensor
import os
import random
import math


class ArcGenerator(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, config):
        super(ArcGenerator, self).__init__()
        self.arc_layer = MultiheadAttention(embed_dim, num_heads, dropout, weights_dropout=False)
        self.arc_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = dropout
        self.config = config

    def forward(self, ids, step, outs, graph_state, graph_padding_mask, attn_mask, strategy_id=None, target_rel=None, work=False):
        x, arc_weight = self.arc_layer(outs, graph_state, graph_state,
                                       key_padding_mask=graph_padding_mask,
                                       attn_mask=attn_mask,
                                       need_weights=True)

        x = F.dropout(x, p=self.dropout, training=self.training)

        if work:
            arc_ll = torch.log(arc_weight+1e-12)
            return arc_ll, outs, x

        target_arc = torch.ne(target_rel, 2) # 0 or 1
        arc_mask = torch.eq(target_rel, 0)

        arc_loss = F.binary_cross_entropy(arc_weight, target_arc.float(), reduction='none')
        arc_loss = arc_loss.masked_fill_(arc_mask, 0.).sum((0, 2))
        arc_loss = arc_loss.mul(strategy_id)
        return arc_loss, outs, x


class ConceptGenerator(nn.Module):
    def __init__(self, embed_dim, dropout, config):
        super(ConceptGenerator, self).__init__()
        self.alignment_layer = MultiheadAttention(embed_dim, 1, dropout, weights_dropout=False)
        self.alignment_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = dropout
        self.config = config
        self.NIL = config.NIL

    def forward(self, ids, step, teacher_force, outs, snt_state, snt_padding_mask, target=None, work=False):
        x, alignment_weight = self.alignment_layer(outs, snt_state, snt_state,
                                                   key_padding_mask=snt_padding_mask,
                                                   need_weights=True)
        x = F.dropout(x, p=self.dropout, training=self.training)
        seq_len, bsz, _ = outs.size()

        copy_probs = alignment_weight.unsqueeze(-1).view(seq_len, bsz, -1)

        ll = torch.log(copy_probs + 1e-12)

        if work:
            return ll

        concept_loss = -ll.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        concept_mask = torch.eq(target, self.NIL)
        concept_loss = concept_loss.masked_fill_(concept_mask, 0.).sum(0)
        return concept_loss

device = torch.device("cuda")

class DecodeLayer(nn.Module):

    def __init__(self, config, inference_layers, num_heads, dropout, ff_embed_dim):
        super(DecodeLayer, self).__init__()
        self.concept_inference_core = Transformer(2, config.hidden_size, ff_embed_dim, num_heads, dropout, with_external = False)
        self.arc_generator = ArcGenerator(config.hidden_size, 1, dropout, config)
        self.concept_generator = ConceptGenerator(config.hidden_size, dropout, config)
        self.position_embeddings = nn.Embedding(config.position_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False)
        self.concept_generator_out = nn.LayerNorm(config.hidden_size)
        self.concept_lstm = nn.LSTM(config.hidden_size, config.hidden_size, 1)
        self.concept_lstm_layer = nn.Linear(config.hidden_size, 256)
        self.concept_fusion_layer = nn.Linear(config.hidden_size + 256, config.hidden_size)
        self.arc_LSTM = nn.LSTM(config.hidden_size, config.hidden_size, 1)
        self.strategy_embeddings = nn.Embedding(3, config.hidden_size)
        self.dropout = dropout

    def forward(self, cls_output, batch_node_embedding, concept_repr, batch_question_embedding,
                batch_node_mask, concept_mask, attn_mask, step=None, new_parent_lists=None, brother_lists=None, concept_in=None, strategy_id=None,
                target=None, target_rel=None, ids=None, work=False):
        if work:
            c_0 = batch_question_embedding
            arc_lstm_output = self.arc_LSTM(concept_repr, (c_0, c_0))
            probe = arc_lstm_output[0][-1, :, :].unsqueeze(0)
            concept_lstm_output = self.concept_lstm(concept_repr, (c_0, c_0))
            concept_lstm_output = self.concept_lstm_layer(concept_lstm_output[0][-1, :, :].unsqueeze(0))
        else:
            c_0 = batch_question_embedding.unsqueeze(0)
            arc_lstm_output = self.arc_LSTM(concept_repr, (c_0, c_0))
            probe = arc_lstm_output[0]
            concept_lstm_output = self.concept_lstm(concept_repr, (c_0, c_0))
            concept_lstm_output = self.concept_lstm_layer(concept_lstm_output[0])

        concept_lstm_output = F.dropout(concept_lstm_output, p=self.dropout, training=self.training)
        outs = F.dropout(probe, p=self.dropout, training=self.training)
        outs = self.layer_norm(outs)

        if work:
            arc_ll, _, x = self.arc_generator(ids, 0, outs, concept_repr, concept_mask, attn_mask, work=True)

            parent = arc_ll.argmax(dim=2).permute(1, 0).cpu().numpy().tolist()
            parent_embedding = torch.zeros(1, concept_repr.shape[1], concept_repr.shape[2]).to("cuda")
            for batch_index in range(concept_repr.size(1)):
                if strategy_id[batch_index] == 1:
                    parent_embedding[:, batch_index, :] = concept_repr[parent[batch_index][0], batch_index, :]
                else:
                    parent_embedding[:, batch_index, :] = concept_repr[-1, batch_index, :]

            parent_embedding = self.layer_norm(parent_embedding)
            outs = self.concept_inference_core(parent_embedding, kv=concept_repr,
                                               self_padding_mask=concept_mask, self_attn_mask=attn_mask)

            strategy_id_reverse = torch.ones(strategy_id.size(0)).to("cuda").masked_fill_(strategy_id.bool(), 0)
            outs = outs * (strategy_id.unsqueeze(0).unsqueeze(2).expand_as(outs).float()) + parent_embedding * (
                strategy_id_reverse.unsqueeze(0).unsqueeze(2).expand_as(outs).float())
            strategy_embeddings = self.strategy_embeddings(strategy_id).unsqueeze(0).expand_as(outs)
            outs = F.dropout(outs, p=self.dropout, training=self.training)
            outs = self.concept_generator_out(self.concept_fusion_layer(torch.cat((concept_lstm_output, outs), dim=2))+strategy_embeddings)
            concept_ll = self.concept_generator(ids, 0, True, outs, batch_node_embedding, batch_node_mask, work=True)
            return concept_ll, arc_ll

        arc_loss, _, x = self.arc_generator(ids, step, outs, concept_repr, concept_mask, attn_mask, strategy_id, target_rel=target_rel, work=False)

        if step < 1:
            teacher_force = True
        else:
            teacher_force = random.random() > step

        if teacher_force:
            parent_embedding = torch.zeros(concept_repr.shape[0], concept_repr.shape[1], concept_repr.shape[2]).to("cuda")
            for batch_index, new_parent_list in enumerate(new_parent_lists):
                for parent_index, parent in enumerate(new_parent_list):
                    parent_embedding[parent_index, batch_index, :] = concept_repr[parent, batch_index, :]

            parent_embedding = self.layer_norm(parent_embedding)
        else:
            parent_embedding = self.layer_norm(x.detach())

        outs = self.concept_inference_core(parent_embedding, kv=concept_repr,
                                           self_padding_mask=concept_mask, self_attn_mask=attn_mask)
        strategy_id_reverse = torch.ones(strategy_id.size(0)).to("cuda").masked_fill_(strategy_id.bool(), 0)
        outs = outs*(strategy_id.unsqueeze(0).unsqueeze(2).expand_as(outs).float()) + parent_embedding*(strategy_id_reverse.unsqueeze(0).unsqueeze(2).expand_as(outs).float())
        strategy_embeddings = self.strategy_embeddings(strategy_id).unsqueeze(0).expand_as(outs)
        outs = F.dropout(outs, p=self.dropout, training=self.training)
        outs = self.concept_generator_out(self.concept_fusion_layer(torch.cat((concept_lstm_output, outs), dim=2)) + strategy_embeddings)
        concept_loss = self.concept_generator(ids, step, teacher_force, outs, batch_node_embedding, batch_node_mask, target=target, work=False)
        concept_tot = concept_mask.size(0) - concept_mask.float().sum(0)
        # node_tot = batch_node_mask.size(0) - batch_node_mask.float().sum(0)
        concept_loss = concept_loss / concept_tot
        arc_loss = arc_loss / concept_tot
        return concept_loss.mean(), arc_loss.mean()


class BertForPRover(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForPRover, self).__init__(config)
        self.config = config
        self.num_labels = 2
        self.strategy_num_labels = 2
        self.inference_layers = 1
        self.num_heads = 8
        self.dropout = 0.1
        self.NIL = config.NIL
        self.END = config.END
        self.DUM = config.DUM
        self.max_node_length = config.max_node_length
        self.ff_embed_dim = config.hidden_size * 2
        self.roberta = RobertaModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.strategy_classifier = nn.Linear(config.hidden_size, self.strategy_num_labels)
        self.naf_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.graph_encoder = Transformer(1, config.hidden_size, self.ff_embed_dim,
                                         self.num_heads, self.dropout, with_external=False)
        self.query_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.self_attn_mask = SelfAttentionMask(device=device)
        self.decoder = DecodeLayer(config, self.inference_layers, self.num_heads, self.dropout, self.ff_embed_dim)
        self.dum_end_embeddings = nn.Embedding(2, config.hidden_size)
        self.rule_fact_embeddings = nn.Embedding(3, config.hidden_size)

        self.position_embeddings = self.decoder.position_embeddings
        self.node_lstm = nn.LSTM(config.hidden_size, config.hidden_size, 1)
        self.concept_lstm = nn.LSTM(config.hidden_size, config.hidden_size, 1)
        self.layer_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False)
        self.embed_scale = math.sqrt(config.hidden_size)

        self.device = config.device
        self.apply(self.init_weights)

    def work(self, batch, beam_size, max_time_step, min_time_step=0):
        with torch.no_grad():
            ids = batch['ids']
            input_ids = batch['input_ids']
            token_type_ids = batch['segment_ids']
            attention_mask = batch['input_mask']
            proof_offset = batch['proof_offset']
            rule_facts_lists = batch['rule_facts_lists']
            questions_contexts = batch['questions_contexts']
            max_sentence_length_list = batch['max_sentence_length_list']
            sentence_count_list = batch['sentence_count_list']
            outputs = self.roberta(input_ids, token_type_ids=None, attention_mask=attention_mask)
            sequence_output = outputs[0]
            cls_output = sequence_output[:, 0, :]
            naf_output = self.naf_layer(cls_output)
            logits = self.classifier(outputs[1])
            logits = logits.argmax(dim=1)
            strategy_logits = self.strategy_classifier(outputs[1])
            strategy_logits = strategy_logits.argmax(dim=1)
            max_node_length = proof_offset.shape[1]
            batch_size = sequence_output.shape[0]
            embedding_dim = sequence_output.shape[2]

            batch_node_embedding = torch.zeros(max_node_length + 3, batch_size, embedding_dim).to("cuda")
            batch_concept_node_embedding = torch.zeros(max_node_length + 3, batch_size, embedding_dim).to("cuda")
            batch_node_mask = torch.zeros(max_node_length + 3, batch_size).bool().to("cuda")
            node_encoder_mask = torch.zeros(max_node_length + 3, batch_size).bool().to("cuda")
            batch_question_embedding = torch.zeros(batch_size, embedding_dim).to("cuda")
            batch_mean_embedding = torch.zeros(batch_size, embedding_dim).to("cuda")

            for batch_index in range(batch_size):
                prev_index = 1
                sample_node_embedding = None
                concept_sample_node_embedding = None
                lstm_input = torch.zeros(max_sentence_length_list[batch_index], sentence_count_list[batch_index],
                                         embedding_dim).to("cuda")
                last_token_index = []
                for proof_offset_index, offset in enumerate(proof_offset[batch_index]):
                    if offset == 0:
                        break
                    else:
                        if proof_offset_index == 0:
                            sentence_embedding = sequence_output[batch_index, prev_index:offset, :]
                            lstm_input[:sentence_embedding.size(0), proof_offset_index, :] = sentence_embedding
                            last_token_index.append(sentence_embedding.size(0) - 1)
                        else:
                            sentence_embedding = sequence_output[batch_index, prev_index:(offset + 2), :]
                            lstm_input[:sentence_embedding.size(0), proof_offset_index, :] = sentence_embedding
                            last_token_index.append(sentence_embedding.size(0) - 1)
                        prev_index = offset + 2

                rule_facts_list = torch.LongTensor(rule_facts_lists[batch_index]).to("cuda")
                rule_facts_embeddings = self.rule_fact_embeddings(rule_facts_list)
                c_0 = rule_facts_embeddings.unsqueeze(0)
                node_lstm_output = self.node_lstm(lstm_input, (c_0, c_0))[0]
                concept_lstm_output = self.concept_lstm(lstm_input, (c_0, c_0))[0]
                for lstm_batch_index in range(node_lstm_output.size(1)):
                    rf_embedding = node_lstm_output[last_token_index[lstm_batch_index], lstm_batch_index, :].unsqueeze(0)
                    concept_rf_embedding = concept_lstm_output[last_token_index[lstm_batch_index], lstm_batch_index, :].unsqueeze(0)
                    if lstm_batch_index == 0:
                        question_embedding = rf_embedding
                        concept_question_embedding = concept_rf_embedding
                    elif lstm_batch_index == 1:
                        sample_node_embedding = rf_embedding
                        concept_sample_node_embedding = concept_rf_embedding
                    else:
                        sample_node_embedding = torch.cat((sample_node_embedding, rf_embedding), dim=0)
                        concept_sample_node_embedding = torch.cat((concept_sample_node_embedding, concept_rf_embedding), dim=0)

                # Add the NAF output at the end
                sample_node_embedding = torch.cat((sample_node_embedding, naf_output[batch_index].unsqueeze(0)), dim=0)
                concept_sample_node_embedding = torch.cat((concept_sample_node_embedding,
                                                           naf_output[batch_index].unsqueeze(0)), dim=0)

                batch_mean_embedding[batch_index, :] = torch.mean(sample_node_embedding, dim=0)
                # Append 0s at the end (these will be ignored for loss)
                node_padding = max_node_length + 3 - sample_node_embedding.shape[0]
                batch_node_mask[:, batch_index] = torch.cat((torch.zeros(sample_node_embedding.shape[0]),
                                                             torch.ones(node_padding)), dim=0).to("cuda")

                node_encoder_mask[:, batch_index] = torch.cat((torch.zeros(sample_node_embedding.shape[0]),
                                                               torch.ones(node_padding)), dim=0).to("cuda")

                rule_facts_list = torch.LongTensor(rule_facts_lists[batch_index]).to("cuda")
                rule_facts_embeddings = self.rule_fact_embeddings(rule_facts_list)

                if strategy_logits[batch_index] == 0:
                    strategy_mask = torch.cat((rule_facts_list, torch.zeros(node_padding).to("cuda")), dim=0).bool()
                    batch_node_mask[:, batch_index] = batch_node_mask[:, batch_index] | (~strategy_mask)

                sample_node_embedding = self.embed_scale * (sample_node_embedding + rule_facts_embeddings)
                sample_node_embedding = torch.cat((sample_node_embedding,
                                                   torch.zeros(node_padding, embedding_dim).to("cuda")), dim=0)

                concept_sample_node_embedding = self.embed_scale * (concept_sample_node_embedding + rule_facts_embeddings)
                concept_sample_node_embedding = torch.cat((concept_sample_node_embedding,
                                                           torch.zeros(node_padding, embedding_dim).to("cuda")), dim=0)

                batch_node_mask[self.END, batch_index] = torch.zeros(1).to("cuda")
                node_encoder_mask[self.DUM, batch_index] = torch.zeros(1).to("cuda")
                node_encoder_mask[self.END, batch_index] = torch.zeros(1).to("cuda")
                batch_node_embedding[:, batch_index, :] = sample_node_embedding
                batch_concept_node_embedding[:, batch_index, :] = concept_sample_node_embedding
                batch_question_embedding[batch_index, :] = question_embedding

            end_embeddings = self.dum_end_embeddings(torch.LongTensor([1]).to("cuda")).expand_as(batch_question_embedding)
            batch_node_embedding[self.DUM, :, :] = batch_question_embedding
            batch_node_embedding[self.END, :, :] = end_embeddings
            batch_node_embedding = self.layer_norm(batch_node_embedding)
            batch_concept_node_embedding[self.DUM, :, :] = batch_question_embedding
            batch_concept_node_embedding[self.END, :, :] = end_embeddings
            batch_concept_node_embedding = self.layer_norm(batch_concept_node_embedding)

            batch_question_embedding = batch_node_embedding[self.DUM, :, :]
            batch_question_embedding = F.dropout(batch_question_embedding, p=self.dropout, training=self.training)
            batch_node_embedding = F.dropout(batch_node_embedding, p=self.dropout, training=self.training)
            batch_concept_node_embedding = F.dropout(batch_concept_node_embedding, p=self.dropout, training=self.training)
            parent_list = torch.tensor([[] for i in range(batch_size)]).to("cuda")
            child_list = torch.tensor([[] for i in range(batch_size)]).to("cuda")
            concept_in = torch.tensor([[] for i in range(batch_size)]).to("cuda")
            mem_dict = {'batch_node_embedding': batch_node_embedding,
                        'batch_concept_node_embedding': batch_concept_node_embedding,
                        'batch_node_mask': batch_node_mask.bool(),
                        'batch_question_embedding': batch_question_embedding.unsqueeze(0),
                        'batch_mean_embedding': batch_mean_embedding.unsqueeze(0),
                        'cls_output': cls_output.unsqueeze(0),
                        'strategy_id': strategy_logits.unsqueeze(0).unsqueeze(2),
                        'batch_parent_lists': parent_list.unsqueeze(0),
                        'batch_child_lists': child_list.unsqueeze(0),
                        'concept_in': concept_in.unsqueeze(0)}
            init_state_dict = {}
            init_hyp = Hypothesis(init_state_dict, [self.DUM], 0., self.END)
            bsz = batch_node_embedding.size(1)
            beams = [Beam(beam_size, min_time_step, max_time_step, [init_hyp], self.END) for i in range(bsz)]
            search_by_batch(self, beams, mem_dict)
        return beams, logits, strategy_logits

    def prepare_incremental_input(self, step_seq):
        conc = torch.from_numpy(ListsToTensor(step_seq, self.NIL)).to("cuda")
        return conc

    def decode_step(self, inp, state_dict, mem_dict, offset, topk):
        step_concept = inp
        batch_node_embedding = mem_dict['batch_node_embedding']
        batch_concept_node_embedding = mem_dict['batch_concept_node_embedding']
        batch_node_mask = mem_dict['batch_node_mask']
        batch_question_embedding = mem_dict['batch_question_embedding']
        cls_output = mem_dict['cls_output']
        strategy_id = mem_dict['strategy_id']
        _, bsz, embedding_dim = batch_node_embedding.size()
        new_state_dict = {}
        concept_repr = torch.zeros((1, bsz, embedding_dim)).to("cuda")
        for index, concept in enumerate(step_concept[0]):
            concept_repr[:, index, :] = torch.index_select(batch_concept_node_embedding[:, index, :], 0, concept)

        name = 'batch_parent_lists'
        if name in state_dict:
            batch_parent_lists = state_dict[name]
        else:
            batch_parent_lists = mem_dict['batch_parent_lists'].squeeze(0)

        name = 'concept_in'
        if name in state_dict:
            concept_in = state_dict[name]
        else:
            concept_in = mem_dict['concept_in'].squeeze(0)
        assert concept_in.size(0) == step_concept.permute(1, 0).size(0)
        concept_in = torch.cat([concept_in, step_concept.permute(1, 0)], 1)

        for idx, layer in enumerate(self.graph_encoder.layers):
            name_i = 'concept_repr_%d'%idx
            if name_i in state_dict:
                prev_concept_repr = state_dict[name_i]
                new_concept_repr = torch.cat([prev_concept_repr, concept_repr], 0)
            else:
                new_concept_repr = concept_repr

            new_state_dict[name_i] = new_concept_repr

            position_ids = torch.arange(new_concept_repr.size(0), dtype=torch.long).to("cuda")
            position_embeddings = self.position_embeddings(position_ids).unsqueeze(1).expand_as(new_concept_repr)
            new_concept_repr = self.layer_norm(new_concept_repr + position_embeddings)
            position_ids = torch.tensor([new_concept_repr.size(0) - 1]).to("cuda")
            position_embeddings = self.position_embeddings(position_ids).unsqueeze(1).expand_as(concept_repr)
            concept_repr = self.layer_norm(concept_repr + position_embeddings)

            # concept_repr, _, _ = layer(concept_repr, kv=new_concept_repr)
        concept_repr = self.layer_norm(concept_repr)
        name = 'graph_state'
        if name in state_dict:
            prev_graph_state = state_dict[name]
            new_graph_state = torch.cat([prev_graph_state, concept_repr], 0)
        else:
            new_graph_state = concept_repr
        new_state_dict[name] = new_graph_state

        query = torch.tanh(self.query_layer(cls_output))
        concept_ll, arc_ll = self.decoder(query, batch_node_embedding, new_graph_state, batch_question_embedding,
                                          batch_node_mask.bool(), None, None, None, batch_parent_lists,
                                          concept_in=concept_in, strategy_id=strategy_id.squeeze(0).squeeze(1), work=True)
        for i in range(offset):
            name = 'arc_ll%d'%i
            new_state_dict[name] = state_dict[name]
        name = 'arc_ll%d'%offset
        new_state_dict[name] = arc_ll * (strategy_id.expand_as(arc_ll).float())

        pred_arc_prob = torch.exp(arc_ll)
        arc_confidence = torch.log(torch.max(pred_arc_prob, 1 - pred_arc_prob))
        arc_confidence[:, :, 0] = 0.
        arc_confidence = arc_confidence * (strategy_id.expand_as(arc_confidence).float())

        LL = concept_ll + arc_confidence.sum(-1, keepdim=True)

        topk_scores, topk_token = torch.topk(LL.squeeze(0), topk, 1)

        new_parent_list = arc_ll.argmax(dim=2).permute(1, 0)
        new_state_dict['batch_parent_lists'] = torch.cat([batch_parent_lists, new_parent_list], 1).int()

        results = []
        for s, t in zip(topk_scores.tolist(), topk_token.tolist()):
            res = []
            for score, token in zip(s, t):
                res.append((token, score))
            results.append(res)

        return new_state_dict, results

    def forward(self, batch, step):
        ids = batch['ids']
        input_ids = batch['input_ids']
        token_type_ids = batch['segment_ids']
        attention_mask = batch['input_mask']
        proof_offset = batch['proof_offset']
        concept_in = batch['concept_in']
        concept_out = batch['concept_out']
        conc = batch['conc']
        rel = batch['rel']
        label_id = batch['label_id']
        strategy_id = batch['strategy_id']
        proofs = batch['proofs']
        component_index_maps = batch['component_index_maps']
        rule_facts_lists = batch['rule_facts_lists']
        new_parent_lists = batch['new_parent_lists']
        new_child_lists = batch['new_child_lists']
        questions_contexts = batch['questions_contexts']
        brother_lists = batch['brother_lists']
        max_sentence_length_list = batch['max_sentence_length_list']
        sentence_count_list = batch['sentence_count_list']
        outputs = self.roberta(input_ids, token_type_ids=None, attention_mask=attention_mask)
        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0, :]
        naf_output = self.naf_layer(cls_output)

        logits = self.classifier(outputs[1])
        loss_fct = CrossEntropyLoss()
        qa_loss = loss_fct(logits.view(-1, self.num_labels), label_id.view(-1))
        strategy_logits = self.strategy_classifier(outputs[1])
        strategy_loss_fct = CrossEntropyLoss()
        strategy_loss = strategy_loss_fct(strategy_logits.view(-1, self.strategy_num_labels), strategy_id.view(-1))
        max_node_length = proof_offset.shape[1]
        batch_size = sequence_output.shape[0]
        embedding_dim = sequence_output.shape[2]

        batch_node_embedding = torch.zeros(max_node_length+3, batch_size, embedding_dim).to("cuda")
        batch_concept_node_embedding = torch.zeros(max_node_length + 3, batch_size, embedding_dim).to("cuda")
        batch_node_mask = torch.zeros(max_node_length+3, batch_size).bool().to("cuda")
        node_encoder_mask = torch.zeros(max_node_length + 3, batch_size).bool().to("cuda")
        batch_question_embedding = torch.zeros(batch_size, embedding_dim).to("cuda")
        batch_mean_embedding = torch.zeros(batch_size, embedding_dim).to("cuda")

        concept_repr = torch.zeros(concept_in.shape[0], concept_in.shape[1], embedding_dim).to("cuda")

        concept_mask = torch.eq(concept_in, self.NIL).to("cuda")
        attn_mask = self.self_attn_mask(concept_in.size(0)).to("cuda")

        for batch_index in range(batch_size):
            prev_index = 1
            sample_node_embedding = None
            concept_sample_node_embedding = None
            lstm_input = torch.zeros(max_sentence_length_list[batch_index], sentence_count_list[batch_index],
                                     embedding_dim).to("cuda")
            last_token_index = []
            for proof_offset_index, offset in enumerate(proof_offset[batch_index]):
                if offset == 0:
                    break
                else:
                    if proof_offset_index == 0:
                        sentence_embedding = sequence_output[batch_index, prev_index:offset, :]
                        lstm_input[:sentence_embedding.size(0), proof_offset_index, :] = sentence_embedding
                        last_token_index.append(sentence_embedding.size(0) - 1)
                    else:
                        sentence_embedding = sequence_output[batch_index, prev_index:(offset + 2), :]
                        lstm_input[:sentence_embedding.size(0), proof_offset_index, :] = sentence_embedding
                        last_token_index.append(sentence_embedding.size(0) - 1)
                    prev_index = offset + 2

            rule_facts_list = torch.LongTensor(rule_facts_lists[batch_index]).to("cuda")
            rule_facts_embeddings = self.rule_fact_embeddings(rule_facts_list)
            c_0 = rule_facts_embeddings.unsqueeze(0)
            node_lstm_output = self.node_lstm(lstm_input, (c_0, c_0))[0]
            concept_lstm_output = self.concept_lstm(lstm_input, (c_0, c_0))[0]
            for lstm_batch_index in range(node_lstm_output.size(1)):
                rf_embedding = node_lstm_output[last_token_index[lstm_batch_index], lstm_batch_index, :].unsqueeze(0)
                concept_rf_embedding = concept_lstm_output[last_token_index[lstm_batch_index], lstm_batch_index, :].unsqueeze(0)
                if lstm_batch_index == 0:
                    question_embedding = rf_embedding
                    concept_question_embedding = concept_rf_embedding
                elif lstm_batch_index == 1:
                    sample_node_embedding = rf_embedding
                    concept_sample_node_embedding = concept_rf_embedding
                else:
                    sample_node_embedding = torch.cat((sample_node_embedding, rf_embedding), dim=0)
                    concept_sample_node_embedding = torch.cat((concept_sample_node_embedding, concept_rf_embedding), dim=0)

            # Add the NAF output at the end
            sample_node_embedding = torch.cat((sample_node_embedding,
                                               naf_output[batch_index].unsqueeze(0)), dim=0)
            concept_sample_node_embedding = torch.cat((concept_sample_node_embedding,
                                                       naf_output[batch_index].unsqueeze(0)), dim=0)

            batch_mean_embedding[batch_index, :] = torch.mean(sample_node_embedding, dim=0)
            # Append 0s at the end (these will be ignored for loss)
            node_padding = max_node_length + 3 - sample_node_embedding.shape[0]
            batch_node_mask[:, batch_index] = torch.cat((torch.zeros(sample_node_embedding.shape[0]),
                                                         torch.ones(node_padding)), dim=0).to("cuda")

            node_encoder_mask[:, batch_index] = torch.cat((torch.zeros(sample_node_embedding.shape[0]),
                                                           torch.ones(node_padding)), dim=0).to("cuda")

            rule_facts_list = torch.LongTensor(rule_facts_lists[batch_index]).to("cuda")
            rule_facts_embeddings = self.rule_fact_embeddings(rule_facts_list)

            if strategy_id[batch_index] == 0:
                strategy_mask = torch.cat((rule_facts_list, torch.zeros(node_padding).to("cuda")), dim=0).bool()
                batch_node_mask[:, batch_index] = batch_node_mask[:, batch_index] | (~strategy_mask)

            sample_node_embedding = self.embed_scale * (sample_node_embedding + rule_facts_embeddings)
            sample_node_embedding = torch.cat((sample_node_embedding,
                                               torch.zeros(node_padding, embedding_dim).to("cuda")), dim=0)

            concept_sample_node_embedding = self.embed_scale * (concept_sample_node_embedding + rule_facts_embeddings)
            concept_sample_node_embedding = torch.cat((concept_sample_node_embedding,
                                                       torch.zeros(node_padding, embedding_dim).to("cuda")), dim=0)

            batch_node_mask[self.END, batch_index] = torch.zeros(1).to("cuda")
            node_encoder_mask[self.DUM, batch_index] = torch.zeros(1).to("cuda")
            node_encoder_mask[self.END, batch_index] = torch.zeros(1).to("cuda")
            batch_node_embedding[:, batch_index, :] = sample_node_embedding
            batch_concept_node_embedding[:, batch_index, :] = concept_sample_node_embedding
            batch_question_embedding[batch_index, :] = question_embedding
        end_embeddings = self.dum_end_embeddings(torch.LongTensor([1]).to("cuda")).expand_as(batch_question_embedding)
        batch_node_embedding[self.DUM, :, :] = batch_question_embedding
        batch_node_embedding[self.END, :, :] = end_embeddings
        batch_node_embedding = self.layer_norm(batch_node_embedding)
        batch_concept_node_embedding[self.DUM, :, :] = batch_question_embedding
        batch_concept_node_embedding[self.END, :, :] = end_embeddings
        batch_concept_node_embedding = self.layer_norm(batch_concept_node_embedding)

        batch_question_embedding = batch_node_embedding[self.DUM, :, :]

        batch_question_embedding = F.dropout(batch_question_embedding, p=self.dropout, training=self.training)
        batch_node_embedding = F.dropout(batch_node_embedding, p=self.dropout, training=self.training)
        batch_concept_node_embedding = F.dropout(batch_concept_node_embedding, p=self.dropout, training=self.training)

        for batch_index in range(batch_size):
            concept_repr[:, batch_index, :] = torch.index_select(batch_concept_node_embedding[:, batch_index, :],
                                                                 0, concept_in[:, batch_index])

        position_ids = torch.arange(concept_repr.size(0), dtype=torch.long).to("cuda")
        position_embeddings = self.position_embeddings(position_ids).unsqueeze(1).expand_as(concept_repr)

        concept_repr = concept_repr + position_embeddings
        concept_repr = self.layer_norm(concept_repr)

        query = torch.tanh(self.query_layer(cls_output))
        concept_loss, arc_loss = self.decoder(query,
                                              batch_node_embedding, concept_repr, batch_question_embedding,
                                              batch_node_mask.bool(), concept_mask.bool(), attn_mask.bool(),
                                              step, new_parent_lists, brother_lists, None, strategy_id,
                                              target=concept_out, target_rel=rel, ids=ids)
        return qa_loss, arc_loss, concept_loss, strategy_loss
