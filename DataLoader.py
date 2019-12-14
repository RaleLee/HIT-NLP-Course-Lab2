import pandas as pd
from pathlib import Path
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from keras_preprocessing import sequence
 
 
def read_Reviews(isTrain=True):
    '''
    To load the reviews from the csv file
    '''
    if isTrain:
        file = "data/Train_reviews.csv"
    else:
        file = "data/Test_labels.csv"
    lines = pd.read_csv(file, sep=",")
    return lines

def read_Labels():
    '''
    To load the labels from the csv file
    '''
    file = "data/Train_labels.csv"
    lines = pd.read_csv(file, sep=",")
    return lines

def text_to_seq(texts, tokenizer, maxlen=48):
    '''
    To turn the text to token
    '''
    input_texts = []
    # turn the text to token
    for idx, text in enumerate(texts):
        text_id_list = [
            tokenizer.vocab.get(token, tokenizer.vocab["[UNK]"]) for token in text
        ]
        input_texts.append(text_id_list)

        assert len(text_id_list) == len(text), "Error found in change text to token"
    
    # do padding
    input_ids = sequence.pad_sequences(
        input_texts, maxlen=maxlen, dtype="long", truncating="post", padding="post"
    )

    # make mask for bert
    masks = np.array([[float(i > 0) for i in _] for _ in input_ids])

    return input_ids, masks

def seq_to_id(maxlen=48):
    '''
    To turn the seq of tokens to seq of ids
    '''
    reviews = read_Reviews()
    labels = read_Labels()
    # reviews and labels both are DataFrame type
    label_Cate_Pola = list( # put all the Categories and Polarities into a list
        labels.apply(
            lambda x: "-".join([x["Categories"], x["Polarities"]]), axis=1
        ).drop_duplicates()  # remove the duplicates
    )

    length_label = len(label_Cate_Pola)
    label_id = dict(
        [(x, y + 1) for x, y in zip(label_Cate_Pola, range(length_label))]
    )

    # tag ids
    # As-B : Begin of Aspect
    # As-M : Middle of Aspect
    # As-E : End of Aspect
    # As-S : Single word of Aspect
    # Op-B : Begin of Opinion
    # Op-M : Middle of Opinion
    # Op-E : End of Opinion
    # Op-S : Single word of Opinion
    tag_id = {"As-B": 1, "As-M": 2, "As-E": 3, "As-S": 4, "Op-B": 5, "Op-M": 6, "Op-E": 7, "Op-S": 8}

    id_label = dict(
        [(v, k) for k, v in label_id.items()]
    )
    id_tag = dict(
        [(v, k) for k, v in tag_id.items()]
    )

    def seq_to_tag(pos, label):
        # turn the seq to tag ids
        seq = np.zeros((maxlen,), dtype=np.int32)
        for start, end in pos:
            if end - start == 1:
                seq[start] = tag_id["%s-S" % label]
            else:
                seq[start] = tag_id["%s-B" % label]
                seq[end-1] = tag_id["%s-E" % label]
                for pos in range(start+1, end-1):
                    seq[pos] = tag_id["%s-M" % label]
        
        return seq.reshape((1, -1))

    def seq_to_label(pos_As, pos_Op, label):
        # turn the seq to labels
        seq = np.zeros((maxlen,), dtype=np.int32)
        for (start, end), lab in zip(pos_As, label):
            if start == " " or int(end) >= maxlen:
                continue
            start = int(start)
            end = int(end)
            if end - start == 1:
                seq[start] = label_id[lab]
            else:
                seq[start] = label_id[lab]
                seq[end-1] = label_id[lab]
                for pos in range(start+1, end-1):
                    seq[pos] = label_id[lab]
        
        for (start, end), lab in zip(pos_Op, label):
            if start == " " or int(end) >= maxlen:
                continue
            start = int(start)
            end = int(end)
            if end - start == 1:
                seq[start] = label_id[lab]
            else:
                seq[start] = label_id[lab]
                seq[end-1] = label_id[lab]
                for pos in range(start+1, end-1):
                    seq[pos] = label_id[lab]

        return seq.reshape((1, -1))
    
    seq_As = labels.groupby("id").apply(
        lambda x: seq_to_tag(
            [
                (int(start), int(end))
                for start, end in zip(x["A_start"], x["A_end"])
                if start != " " and int(end) < maxlen
            ],
            "As"
        )
    )

    seq_Op = labels.groupby("id").apply(
        lambda x: seq_to_tag(
            [
                (int(start), int(end))
                for start, end in zip(x["O_start"], x["O_end"])
                if start != " " and int(end) < maxlen
            ],
            "Op"
        )
    )

    seq_Cate_Pola = labels.groupby("id").apply(
        lambda x: seq_to_label(
            [(start, end) for start, end in zip(x["A_start"], x["A_end"])],
            [(start, end) for start, end in zip(x["O_start"], x["O_end"])],
            [emo+"-"+pos for emo, pos in zip(x["Categories"], x["Polarities"])]
        )
    )

    seq_id = np.array(
        labels.groupby("id").apply(
            lambda x: list(x["id"])[0]
        ).to_list()
    )
    seq_As = np.vstack(seq_As)
    seq_Op = np.vstack(seq_Op)
    seq_As_Op = seq_As + seq_Op 
    seq_Cate_Pola = np.vstack(seq_Cate_Pola)

    return seq_id, seq_As_Op, seq_Cate_Pola, id_label, id_tag

def pair_As_Op(As_tags, Op_tags):
    views = []
    tmp = []
    for idx_As, As_tag in enumerate(As_tags):
        middle_As = (As_tag[2] + As_tag[3]) / 2
        dis = []
        
        if len(Op_tags) == 0:
            break
        for o_t in Op_tags:
            middle_o = (o_t[2] + o_t[3]) / 2
            di = abs(middle_As - middle_o) + (As_tag[1] != o_t[1]) * 5
            dis.append(di)
        idx_Op = np.array(dis).argmin()

        if As_tag[1] == Op_tags[idx_Op][1]:
            views.append((As_tag[0], Op_tags[idx_Op][0], As_tag[1]))
            del Op_tags[idx_Op]
            tmp.append(idx_As)
    
    for o_t in Op_tags:
        views.append(("_", o_t[0], o_t[1]))

    for idx_As, As_tag in enumerate(As_tags):
        if idx_As in tmp:
            continue
        views.append((As_tag[0], "_", As_tag[1]))
    
    return views

def seq_to_word(seq_id, seq_As_Op, seq_Cate_Pola, id_label, id_tag, text_review):
    '''
    To turn the encoded seq to word
    '''
    assert seq_As_Op.shape[0] == seq_Cate_Pola.shape[0] == len(text_review)
    maxlen = seq_As_Op.shape[1]
    seq_idx = np.arange(maxlen)

    views = []

    for idx, s_ao, s_cp, text in zip(seq_id, seq_As_Op, seq_Cate_Pola, text_review):
        idx_ab = seq_idx[np.where(s_ao == 1, True, False)]
        idx_am = seq_idx[np.where(s_ao == 4, True, False)]
        idx_ae = seq_idx[np.where(s_ao == 3, True, False)]
        idx_ob = seq_idx[np.where(s_ao == 5, True, False)]
        idx_om = seq_idx[np.where(s_ao == 8, True, False)]
        idx_oe = seq_idx[np.where(s_ao == 7, True, False)]
        
        a_tags, o_tags = [], []
        for i_b, i_e in zip(idx_ab, idx_ae):
            if i_b >= i_e + 1:
                continue
            label = max(s_cp[i_b:i_e+1])
            a_tags.append((text[i_b:i_e+1], label, i_b, i_e+1))
        
        for i_m in idx_am:
            label = max(s_cp[i_m: i_m + 1])
            a_tags.append((text[i_m : i_m+1], label, i_m, i_m+1))

        for i_b, i_e in zip(idx_ob, idx_oe):
            if i_b >= i_e + 1:
                continue
            label = max(s_cp[i_b:i_e+1])
            o_tags.append((text[i_b:i_e+1], label, i_b, i_e+1))
        
        for i_m in idx_om:
            label = max(s_cp[i_m: i_m + 1])
            o_tags.append((text[i_m : i_m+1], label, i_m, i_m+1))
        
        vp = pair_As_Op(a_tags.copy(), o_tags.copy())
        vp = [(str(idx), ) + v[:2] + tuple(id_label[v[2]].split("-")) for v in vp if v[2] > 0]
        views.extend(vp)

        if len(a_tags) > 0 and len(a_tags) < len(o_tags):
            pass
        if len(a_tags) > 0 and len(a_tags) == len(o_tags):
            pass    
        if len(idx_am) > 0 or len(idx_om) > 0:
            pass
        if len(a_tags) > 0 and len(a_tags) > len(o_tags):
            pass

    return views    