import sys
import os
import csv


def prepare_data(file):
    map_data = {}
    with open(file, "r") as f:
        data = f.readline()
        key = ""
        exit_flag = False
        while data:
            data = data.strip()
            if data == "":
                if exit_flag:
                    break
                exit_flag = True
            else:
                exit_flag = False
            if ".csv" in data:
                data = data.split("=")
                key = data[1][1:-1]
                map_data[key] = []
            else:
                if data != "":
                    map_data[key].append(data)
            data = f.readline()
    return map_data


def evaluate(directory, ds):
    total = 0
    correct = 0
    for key, values in ds.items():
        with open(os.path.join(directory, key)) as file:
            csv_reader = list(csv.DictReader(file))
            total += len(csv_reader)
            for i in range(len(csv_reader)):
                if values[i] == csv_reader[i]["act_tag"]:
                    correct += 1
    return (correct/total)*100


if __name__ == "__main__":
    ds = prepare_data(sys.argv[2])
    result = evaluate(sys.argv[1], ds)
    print(result)

