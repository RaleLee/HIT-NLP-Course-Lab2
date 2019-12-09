import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from pytorch_pretrained_bert.modeling import BertModel

import os
import numpy as np

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from DataLoader import *
from models import TokenNet, JointSentCateNet, UnionNet
from convert_phase import convert

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Processor(object):
    def __init__(self, args):
        self.__max_len = args.max_len
        self.__max_len_sent = args.max_len_sent
        self.__batch_size = args.batch_size
        self.__test_size = args.test_size
        self.__bert_path = args.bert_path
        self.__lr = args.learning_rate
        self.__dropout_rate = args.dropout_rate
        self.__save_dir = args.save_dir
        self.__output_dir = args.output_dir
        self.__bert_embedding_size = args.bert_embedding_size
        self.__pola_class = args.pola_class
        self.__cate_class = args.cate_class
        self.__tag_class = 9
        self.__n_class = 38

    def split_viewpoints(self, seq_id, seq_input, seq_mask, seq_AO, seq_CP):

        idx = np.random.permutation(range(seq_id.shape[0]))
        tr_idx, te_idx = train_test_split(idx, test_size=self.__test_size)

        return (
            (
                seq_id[tr_idx],
                seq_input[tr_idx],
                seq_mask[tr_idx],
                seq_AO[tr_idx],
                seq_CP[tr_idx],
            ),
            (
                seq_id[te_idx],
                seq_input[te_idx],
                seq_mask[te_idx],
                seq_AO[te_idx],
                seq_CP[te_idx],
            ),
        )


    def make_viewloader(self, xy_tr, xy_te, bs=16):

        tr_id, tr_input, tr_mask, tr_AO, tr_CP = xy_tr
        te_id, te_input, te_mask, te_AO, te_CP = xy_te

        tr_id = torch.LongTensor(tr_id)
        te_id = torch.LongTensor(te_id)

        tr_input = torch.LongTensor(tr_input)
        te_input = torch.LongTensor(te_input)

        tr_mask = torch.LongTensor(tr_mask)
        te_mask = torch.LongTensor(te_mask)

        tr_AO = torch.LongTensor(tr_AO)
        te_AO = torch.LongTensor(te_AO)

        tr_CP = torch.LongTensor(tr_CP)
        te_CP = torch.LongTensor(te_CP)

        train_data = TensorDataset(tr_id, tr_input, tr_mask, tr_AO, tr_CP)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

        valid_data = TensorDataset(te_id, te_input, te_mask, te_AO, te_CP)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

        return train_dataloader, valid_dataloader


    def get_opt(self, model, finetune=True):

        if finetune:
            param_optimizer = list(model.named_parameters())
            no_decay = ["bias", "gamma", "beta"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay_rate": 0.01,
                },
                {
                    "params": [
                        p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay_rate": 0.0,
                },
            ]
        else:
            param_optimizer = list(model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)
        # optimizer = Adam(model.parameters(), lr=0.0001)
        return optimizer


    def train_model(self, model, view_trloader):
        max_grad_norm = 1.0
        optimizer = self.get_opt(model, True)
        # criterion = nn.CrossEntropyLoss(ignore_index=0)
        criterion = nn.CrossEntropyLoss()

        # TRAIN loop
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        pbar = tqdm(view_trloader)  
        for step, batch in enumerate(pbar):
            # add batch to gpu

            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            b_id, b_input, b_mask, b_AO, b_CP = batch

            logit_AO, logit_CP = model(b_input, b_mask)

            loss = criterion(logit_AO.view(-1, logit_AO.shape[-1]), b_AO.view(-1))
            loss += criterion(logit_CP.view(-1, logit_CP.shape[-1]), b_CP.view(-1))

            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input.size(0)
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=max_grad_norm
            )
            # update parameters
            optimizer.step()

            desc = "loss-%.3f" % (tr_loss / nb_tr_steps)
            pbar.set_description(desc)


    def test_model(self, model, view_trloader):
        max_grad_norm = 1.0
        criterion = nn.CrossEntropyLoss()


        model.eval()
        eval_loss = 0
        nb_eval_examples, nb_eval_steps = 0, 0
        pbar = tqdm(view_trloader)  
        
        seq_id, seq_AO, seq_CP = [], [], []
        
        for step, batch in enumerate(pbar):
            # add batch to gpu

            batch = tuple(t.to(device) for t in batch)
            b_id, b_input, b_mask, b_AO, b_CP = batch

            with torch.no_grad():
                logit_AO, logit_CP = model(b_input, b_mask)

            tmp_eval_loss = criterion(logit_AO.view(-1, logit_AO.shape[-1]), b_AO.view(-1))
            tmp_eval_loss += criterion(logit_CP.view(-1, logit_CP.shape[-1]), b_CP.view(-1))

            s_ao = logit_AO.detach().cpu().numpy()
            s_cp = logit_CP.detach().cpu().numpy()
            seq_id.extend(b_id.detach().cpu().numpy())
            seq_AO.append(np.argmax(s_ao, axis=2))
            seq_CP.append(np.argmax(s_cp, axis=2))
        
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_examples += b_input.size(0)
            nb_eval_steps += 1

            desc = "loss-%.3f" % (eval_loss / nb_eval_steps)
            pbar.set_description(desc)

        seq_AO = np.vstack(seq_AO)
        seq_CP = np.vstack(seq_CP)
        return seq_id, seq_AO, seq_CP


        
    def generate_result(self, model, texts, seq_id, tokenizer, id_to_label, id_to_term):
        
        seq_AO, seq_CP = [], []
        seq_input, seq_mask = text_to_seq(texts,tokenizer,maxlen=60)
        b_input = torch.LongTensor(seq_input)
        b_mask = torch.LongTensor(seq_mask)
        b_input = b_input.to(device)
        b_mask = b_mask.to(device)
        
        logit_AO, logit_CP = model(b_input, b_mask)
        s_ao = logit_AO.detach().cpu().numpy()
        s_cp = logit_CP.detach().cpu().numpy()
        seq_AO.append(np.argmax(s_ao, axis=2))
        seq_CP.append(np.argmax(s_cp, axis=2))
        

        seq_AO = np.vstack(seq_AO)
        seq_CP = np.vstack(seq_CP)
        pred_vp = seq_to_word(
                seq_id, seq_AO, seq_CP, id_to_label, id_to_term, texts)
        
        return pred_vp

    def cal_metrics(self, pred_vp, true_vp):

        
        p = len(pred_vp)
        g = len(true_vp)
        s = len(set(pred_vp) & set(true_vp))
            
        precision = s / (p+0.00001)
        recall = s / (g+0.000001)
        f1 = 2 * precision * recall / (precision + recall + 0.00001)
        print(
            precision,
            recall,
            f1,
            "\n",
        )
        
        return f1

    def train(self):
        df_review = read_Reviews(True)
        df_label = read_Labels()
        df_review_test = read_Reviews(False)
        seq_id, seq_AO, seq_CP, id_to_label, id_to_term = seq_to_id(maxlen=self.__max_len)
        tokenizer = BertTokenizer.from_pretrained(self.__bert_path, do_lower_case=True)
        seq_input, seq_mask = text_to_seq(
            list(df_review["Reviews"]), tokenizer, maxlen=self.__max_len
        )

        true_vp = [
            (
                str(row["id"]),
                row["AspectTerms"],
                row["OpinionTerms"],
                row["Categories"],
                row["Polarities"],
            )
            for rowid, row in df_label.iterrows()
        ]
            
        pred_vp = seq_to_word(
            seq_id, seq_AO, seq_CP, id_to_label, id_to_term, list(df_review["Reviews"])
        )

        self.cal_metrics(pred_vp, true_vp)

        view_tr, view_te = self.split_viewpoints(seq_id, seq_input, seq_mask, seq_AO, seq_CP)

        train_loader, test_loader = self.make_viewloader(view_tr, view_te, bs=self.__batch_size)

        bert_base = BertModel.from_pretrained(self.__bert_path)
        model_view = TokenNet(bert_base, self.__bert_embedding_size, self.__dropout_rate, self.__tag_class, self.__n_class, finetuning=True)

        path_model_bert = os.path.join(self.__save_dir, "bert_best.pt")
        if os.path.exists(path_model_bert):
            res = torch.load(path_model_bert)
            bert_base.load_state_dict(res["bert_base"])

        path_model_view = os.path.join(self.__save_dir, "model_view.pt")
        if os.path.exists(path_model_view):
            res = torch.load(path_model_view)
            model_view.load_state_dict(res["model_view"])

        model_view.to(device)

        max_f1 = 0.7
        for epoch in range(100):

            self.train_model(model_view, train_loader)
            seq_id, seq_AO, seq_CP = self.test_model(model_view, test_loader)
            
            true_vp = [
            (
                str(row["id"]),
                row["AspectTerms"],
                row["OpinionTerms"],
                row["Categories"],
                row["Polarities"],
            )
            for rowid, row in df_label[df_label['id'].isin(seq_id)].iterrows()
            ]
            
            texts = [df_review[df_review['id']==i]["Reviews"].values[0] for i in seq_id]
            pred_vp = seq_to_word(
                seq_id, seq_AO, seq_CP, id_to_label, id_to_term, texts)
            
            f1 = self.cal_metrics(pred_vp, true_vp)
            
            if f1 > max_f1:
                state = {"bert_base": bert_base.state_dict()}
                torch.save(state, os.path.join(self.__save_dir, "bert_best.pt"))
                state = {"model_view": model_view.state_dict()}
                torch.save(state, os.path.join(self.__save_dir, "model_view.pt"))
                
                max_f1 = f1

                lines = []
                for rowid, row in tqdm(df_review_test.iterrows()):
                    pred_vp = self.generate_result(model_view, [row['Reviews']], [row['id']], tokenizer, id_to_label, id_to_term)
                    if len(pred_vp) == 0:
                        pred_vp = [(str(rowid), "_", "_", "_", "_")]

                    for vp in pred_vp:
                        line = ",".join(vp)
                        lines.append(line)

                    with open(self.__output_dir + ".csv", "w") as f:
                        for line in lines:
                            f.write(line + "\n")
        
        # to change the result to phase answer
        phase1path = self.__output_dir + "task1_answer.csv"
        phase2path = self.__output_dir + "task2_answer.csv"
        phase3path = self.__output_dir + "task3_answer.csv"
        ori = self.__output_dir + ".csv"
        convert(phase1path, phase2path, phase3path, ori)

            
