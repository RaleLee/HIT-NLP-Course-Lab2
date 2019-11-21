import csv

def convert(ph1, ph2, ph3):
    
    with open(ph3, 'r', encoding='utf-8') as f:
        with open(ph1, 'w', newline='', encoding='utf-8') as p1:
            with open(ph2, 'w', newline='', encoding='utf-8') as p2:
                csv_reader = csv.reader(f)
                p1_writer = csv.writer(p1)
                p2_writer = csv.writer(p2)
                data = []
                for row in csv_reader:
                    p1_writer.writerow(row[:-2])
                    p2_writer.writerow(row[:-1])
            p2.close()
        p1.close()
    f.close()
    return

if __name__ == "__main__":
    phase1path = "outputs/phase1_1121.csv"
    phase2path = "outputs/phase2_1121.csv"
    phase3path = "outputs/pred_1121.csv"
    convert(phase1path, phase2path, phase3path)
