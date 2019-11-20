# encoding = utf-8
# index B-As, I-As, B-Op, I-Op, O
# each row in csv file is a list
import csv

def load_reviews(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        for row in csv_reader:
            data.append(row)
    f.close()
    return data

def load_labels(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        for row in csv_reader:
            if len(row) == 0:
                continue
            idx = row[0]
            if idx not in data:
                data[idx] = []
                data[idx].append(row[1:])
            else:
                data[idx].append(row[1:])
    
    f.close()
    return data

def write_labels_test(reviews, path, data):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # writer.writerow([header])
        len_r = len(reviews)
        len_data = len(len_data)
        assert len_r == len_data, "Something wrong!"
        labels = {}
        for j in range(len_r):
            labels[j] = []
            length = len(data[j])
            line = data[j]
            i = 0
            while i < length:
                if line[i] == 0:
                    bas = i
                    i += 1
                    while i < length and line[i] == 1:
                        i += 1
                    eas = i
                    labels[j].append((bas, eas))
                    continue
                if line[i] == 2:
                    bop = i
                    i += 1
                    while i < length and line[i] == 3:
                        i += 1
                    eop = i
                    labels[j].append((bop, eop))
                    continue
                if line[i] == 4:
                    i += 1
                    continue
        
        write_label = []
        for j in range(len_r):
            label = labels[j]
            review = reviews[j]
            for be, ed in label:
                li = review[be, ed]
                st = ""
                for ch in li:
                    st += ch
                wri = [str(j), st]
                write_label.append(wri)

        for line in write_label:
            writer.writerow(line)
    f.close()



def loadTrainingData_Phase1(review_path, label_path):
    training_data = []
    reviews = load_reviews(review_path)
    labels = load_labels(label_path)

    for review in reviews:
        idx = review[0]
        content = review[1]
        words = []
        for ch in content:
            words.append(ch)
        
        # get index labels
        las = labels[idx]
        length = len(words)
        label = []
        for i in range(length):
            label.append("O")
        
        for la in las:
            # aspect label
            if not la[0] == "_":
                label[int(la[1])] = "B-As"
                for i in range(int(la[1]) + 1, int(la[2])):
                    label[i] = "I-As"
            
            # opinion label
            if not la[3] == "_":
                label[int(la[4])] = "B-Op"
                for i in range(int(la[4]) + 1, int(la[5])):
                    label[i] = "I-Op"
        
        line = (words, label)
        training_data.append(line)


    return training_data

def loadTestData_Phase1(test_path):
    test_data = []
    reviews = load_reviews(test_path)

    for review in reviews:
        idx = review[0]
        content = review[1]
        words = []
        for ch in content:
            words.append(ch)
        
        test_data.append(words)
    
    return test_data

