import sys
import os
import csv
import pycrfsuite
import nltk

x_train = []
y_train = []
map_features = {}


def process_features(directory, predict_flag=False):
    for root, subdir, files in os.walk(directory):
        i = 0
        for f in files:
            # if i == 20:
            #     break
            i += 1
            file_record_list = []
            file_act_tag_list = []
            with open(os.path.join(root, f)) as file:
                csv_reader = csv.DictReader(file)
                first_record = True
                speaker = ''
                for map_csv in csv_reader:
                    record_list = []
                    if map_csv["speaker"] != speaker:
                        speaker = map_csv["speaker"]
                        record_list.append("SC")

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
                        bigrams = nltk.ngrams(map_csv["pos"].split(), 2)
                        for pos in bigrams:
                            zero_token, zero_tag = pos[0].split('/')
                            one_token, one_tag = pos[1].split('/')
                            record_list.append("TOKEN_" + zero_token + "|" + one_token)
                            record_list.append("TOKEN_" + zero_tag + "|" + one_tag)
                        # trigrams = nltk.ngrams(map_csv["pos"].split(), 3)
                        # for pos in trigrams:
                        #     zero_token, zero_tag = pos[0].split('/')
                        #     one_token, one_tag = pos[1].split('/')
                        #     two_token, two_tag = pos[2].split('/')
                        #     record_list.append("TOKEN_" + zero_token + "|" + one_token + "|" + two_token)
                        #     record_list.append("TOKEN_" + zero_tag + "|" + one_tag+ "|" + two_tag)
                    if map_csv["act_tag"]:
                        file_act_tag_list.append(map_csv["act_tag"])
                    else:
                        file_act_tag_list.append(" ")
                    file_record_list.append(record_list)

            if len(file_record_list) != 0 and len(file_act_tag_list) != 0:
                if predict_flag:
                    map_features[f] = [file_record_list, file_act_tag_list]
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
    total = 0
    correct = 0
    with open(op_file, 'w') as file:
        for name, features in map_features.items():
            tags = tagger.tag(features[0])
            file.write("Filename=\"" + name + "\"\n")
            for i in range(len(tags)):
                total += 1
                if features[1][i] == tags[i]:
                    correct += 1
                file.write(tags[i] + "\n")
            file.write("\n")
    return correct, total


if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    process_features(sys.argv[1])
    train()
    process_features(sys.argv[2], predict_flag=True)
    correct, total = predict(sys.argv[3])
    print("#" * 70)
    print((correct/total) * 100)
    print(str(timeit.default_timer() - start))
