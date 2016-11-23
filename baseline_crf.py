import sys
import os
import csv
import pycrfsuite

x_train = []
y_train = []
map_features = {}


def process_features(directory, predict_flag=False):
    for root, subdir, files in os.walk(directory):
        for f in files:
            file_record_list = []
            file_act_tag_list = []
            with open(os.path.join(root, f)) as file:
                if ".csv" not in file.name:
                    continue
                csv_reader = csv.DictReader(file)
                first_record = True
                speaker = ''
                for map_csv in csv_reader:
                    record_list = []
                    if first_record:
                        record_list.append("FU")
                        speaker = map_csv["speaker"]
                        first_record = False

                    if map_csv["speaker"] != speaker:
                        speaker = map_csv["speaker"]
                        record_list.append("SC")

                    if map_csv["pos"]:
                        for pos in map_csv["pos"].split():
                            token, tag = pos.split("/")
                            record_list.append("TOKEN_" + token)
                            record_list.append("POS_" + tag)
                    if map_csv["act_tag"]:
                        file_act_tag_list.append(map_csv["act_tag"])

                    file_record_list.append(record_list)

            if len(file_record_list) != 0 and len(file_act_tag_list) != 0:
                if predict_flag:
                    map_features[f] = file_record_list
                else:
                    x_train.append(file_record_list)
                    y_train.append(file_act_tag_list)


def train():
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(x_train, y_train):
        trainer.append(xseq, yseq)
    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    trainer.train('train.crfsuite')


def predict(op_file):
    tagger = pycrfsuite.Tagger()
    tagger.open('train.crfsuite')
    with open(op_file, 'w') as file:
        for name, file_list in map_features.items():
            tag_list = tagger.tag(file_list)
            file.write("Filename=\"" + name + "\"\n")
            for tag in tag_list:
                file.write(tag + "\n")
            file.write("\n")


if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    process_features(sys.argv[1])
    train()
    process_features(sys.argv[2], predict_flag=True)
    predict(sys.argv[3])
    print()
