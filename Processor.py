import torch
import torch.nn as nn
import os
import numpy as np
from torch.optim import Adam
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from pytorch_pretrained_bert.modeling import BertModel
from tqdm import tqdm

from Dataloader import *
from utils import *
from convert_phase import convert
from model import JointAsOpCaPoNet

class Processor(object):

    def __init__(self, args):

        # init parameters part
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
        self.__epoch = args.epoch
        self.__tag_class = 9
        self.__n_class = 38
        
        # init models
        self.__bert_base = BertModel.from_pretrained(self.__bert_path)
        self.__model = JointAsOpCaPoNet(
            self.__bert_base,
            self.__bert_embedding_size,
            self.__dropout_rate,
            self.__tag_class,
            self.__n_class,
            finetuning=True
        )
        self.__device = "cpu"
        if torch.cuda.is_available():
            self.__device = "cuda"
        
        self.__model.to(self.__device)
 
    def get_opt(self, finetune=True):

        if finetune:
            param_optimizer = list(self.__model.named_parameters())
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
            param_optimizer = list(self.__model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        optimizer = Adam(optimizer_grouped_parameters, lr=self.__lr)
        return optimizer


    def train_model(self, view_trloader):
        max_grad_norm = 1.0
        optimizer = self.get_opt(True)
        criterion = nn.CrossEntropyLoss()

        # TRAIN loop
        self.__model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        pbar = tqdm(view_trloader)  
        for step, batch in enumerate(pbar):
            optimizer.zero_grad()
            batch = tuple(t.to(self.__device) for t in batch)
            b_id, b_input, b_mask, b_AO, b_CP = batch

            logit_AO, logit_CP = self.__model(b_input, b_mask)

            loss = criterion(logit_AO.view(-1, logit_AO.shape[-1]), b_AO.view(-1))
            loss += criterion(logit_CP.view(-1, logit_CP.shape[-1]), b_CP.view(-1))

            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input.size(0)
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=self.__model.parameters(), max_norm=max_grad_norm
            )
            # update parameters
            optimizer.step()

            desc = "loss-%.3f" % (tr_loss / nb_tr_steps)
            pbar.set_description(desc)


    def test_model(self, view_trloader):
        max_grad_norm = 1.0
        criterion = nn.CrossEntropyLoss()


        self.__model.eval()
        eval_loss = 0
        nb_eval_examples, nb_eval_steps = 0, 0
        pbar = tqdm(view_trloader)  
        
        seq_id, seq_AO, seq_CP = [], [], []
        
        for step, batch in enumerate(pbar):
            # add batch to gpu

            batch = tuple(t.to(self.__device) for t in batch)
            b_id, b_input, b_mask, b_AO, b_CP = batch

            with torch.no_grad():
                logit_AO, logit_CP = self.__model(b_input, b_mask)

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


        
    def generate_result(self, texts, seq_id, tokenizer, id_to_label, id_to_term):
        
        seq_AO, seq_CP = [], []
        seq_input, seq_mask = text_to_seq(texts,tokenizer,maxlen=60)
        b_input = torch.LongTensor(seq_input)
        b_mask = torch.LongTensor(seq_mask)
        b_input = b_input.to(self.__device)
        b_mask = b_mask.to(self.__device)
        
        logit_AO, logit_CP = self.__model(b_input, b_mask)
        s_ao = logit_AO.detach().cpu().numpy()
        s_cp = logit_CP.detach().cpu().numpy()
        seq_AO.append(np.argmax(s_ao, axis=2))
        seq_CP.append(np.argmax(s_cp, axis=2))
        

        seq_AO = np.vstack(seq_AO)
        seq_CP = np.vstack(seq_CP)
        pred = seq_to_word(seq_id, seq_AO, seq_CP, id_to_label, id_to_term, texts)
        
        return pred

    def train(self):
        review = read_Reviews(True)
        label = read_Labels()
        review_test = read_Reviews(False)
        seq_id, seq_AO, seq_CP, id_to_label, id_to_term = seq_to_id(maxlen=self.__max_len)
        tokenizer = BertTokenizer.from_pretrained(self.__bert_path, do_lower_case=True)
        seq_input, seq_mask = text_to_seq(
            list(review["Reviews"]), tokenizer, maxlen=self.__max_len
        )

        true_l = [
            (
                str(row["id"]),
                row["AspectTerms"],
                row["OpinionTerms"],
                row["Categories"],
                row["Polarities"],
            )
            for rowid, row in label.iterrows()
        ]
            
        pred = seq_to_word(
            seq_id, seq_AO, seq_CP, id_to_label, id_to_term, list(review["Reviews"])
        )

        cal_metrics(pred, true_l)

        view_tr, view_te = split_train_test(self.__test_size, seq_id, seq_input, seq_mask, seq_AO, seq_CP)

        train_loader, test_loader = dataloader(view_tr, view_te, batchSize=self.__batch_size)

        path_model_bert = os.path.join(self.__save_dir, "bert_best.pt")
        if os.path.exists(path_model_bert):
            res = torch.load(path_model_bert)
            self.__bert_base.load_state_dict(res["bert_base"])

        path_model = os.path.join(self.__save_dir, "model.pt")
        if os.path.exists(path_model):
            res = torch.load(path_model)
            self.__model.load_state_dict(res["model"])

        max_f1 = 0.7
        for epoch in range(self.__epoch):
            print("Epoch" + str(epoch) + ":")
            self.train_model(train_loader)
            seq_id, seq_AO, seq_CP = self.test_model(test_loader)
            
            true_l = [
            (
                str(row["id"]),
                row["AspectTerms"],
                row["OpinionTerms"],
                row["Categories"],
                row["Polarities"],
            )
            for rowid, row in label[label['id'].isin(seq_id)].iterrows()
            ]
            
            texts = [review[review['id']==i]["Reviews"].values[0] for i in seq_id]
            pred = seq_to_word(
                seq_id, seq_AO, seq_CP, id_to_label, id_to_term, texts)
            
            f1 = cal_metrics(pred, true_l)
            
            if f1 > max_f1:
                state = {"bert_base": self.__bert_base.state_dict()}
                torch.save(state, os.path.join(self.__save_dir, "bert_best.pt"))
                state = {"model": self.__model.state_dict()}
                torch.save(state, os.path.join(self.__save_dir, "model.pt"))
                
                max_f1 = f1

                lines = []
                for rowid, row in tqdm(review_test.iterrows()):
                    pred = self.generate_result([row['Reviews']], [row['id']], tokenizer, id_to_label, id_to_term)
                    if len(pred) == 0:
                        pred = [(str(rowid), "_", "_", "_", "_")]

                    for vp in pred:
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

            
