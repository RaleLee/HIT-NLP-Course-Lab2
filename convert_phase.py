import csv

def convert(ph1, ph2, ph3, ori):
    
    with open(ori, 'r', encoding='utf-8') as f:
        with open(ph3, 'w',newline='', encoding='utf-8') as p3:
            with open(ph1, 'w', newline='', encoding='utf-8') as p1:
                with open(ph2, 'w', newline='', encoding='utf-8') as p2:
                    csv_reader = csv.reader(f)
                    p1_writer = csv.writer(p1)
                    p2_writer = csv.writer(p2)
                    p3_writer = csv.writer(p3)
                    data = []
                    for row in csv_reader:
                        p1_writer.writerow(row[:-2])
                        p2_writer.writerow(row[:-1])
                        p3_writer.writerow(row)
                p2.close()
            p1.close()
        p3.close()
    f.close()

    return

if __name__ == "__main__":
    phase1path = "outputs/task1_answer.csv"
    phase2path = "outputs/task2_answer.csv"
    phase3path = "outputs/task3_answer.csv"
    oripath = "outputs/result.csv"
    convert(phase1path, phase2path, phase3path, oripath)
