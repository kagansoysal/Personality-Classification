import pandas as pd
import numpy as np

df = pd.read_csv("16P.csv", encoding="ISO-8859-1")
df.drop(["Response Id"], inplace=True, axis=1)
df_without_normalization = pd.DataFrame(df)
df.replace([-3, -2, -1, 0, 1, 2, 3], [0, 1/6, 2/6, 3/6, 4/6, 5/6, 1], inplace=True) # We are applying normalization to one of our DataFrames
personalities = ["ESTJ", "ENTJ", "ESFJ", "ENFJ", "ISTJ", "ISFJ", "INTJ", "INFJ", "ESTP", "ESFP", "ENTP", "ENFP", "ISTP",
                 "ISFP", "INTP", "INFP"]
df_without_normalization.replace(personalities, range(16), inplace=True)
df.replace(personalities, range(16), inplace=True)
performance = {}
(int1, int2, int3, int4, int5, int6) = (0, 12000, 24000, 36000, 48000, 59999)
intervals = list(zip((int1, int2, int3, int4, int5), (int2, int3, int4, int5, int6))) # Since we will divide our test by 5, we create our intervals
number_true = 0
number_false = 0

def k_fold_cross_validation(k_value, interval):
    global number_true, number_false

    metrics = ["TP", "TN", "FP", "FN"]
    for personality in personalities: # We create this dictionary to hold the metric values for each class
        performance[personality] = {}
        for character in performance:
            for metric in metrics:
                performance[character][metric] = 0 #
    for test_person in range(interval[0], interval[1]):
        test_sample = datas[test_person, :-1]
        train_samples = np.concatenate((datas[:interval[0], :-1], datas[interval[1]:int6, :-1]))
        substraction = test_sample - train_samples
        squared_substraction = substraction ** 2
        sum_squared_substraction = np.sum(squared_substraction, axis=1)
        distances = sum_squared_substraction ** (1 / 2) # We calculated the distances
        train_interval = list(range(interval[0])) + list(range(interval[1], int6))
        dict_distances = dict(zip(distances, train_interval)) # We matched which distance belongs to which player.
        distances.sort()
        nearest_neighbors = [dict_distances[distances[k]] for k in range(k_value)]
        nearest_personalities = [datas[index, -1] for index in nearest_neighbors]
        most = max(set(nearest_personalities), key=nearest_personalities.count) # According to the k value given, we estimated which personality has the most among the nearest neighbors for the test person.

        # In the following part, we calculate the metric values for each class
        if most == datas[test_person, -1]:
            number_true += 1
            performance[personalities[int(datas[test_person, -1])]]["TP"] += 1
            for personality in performance:
                if personality == personalities[int(datas[test_person, -1])]:
                    continue
                else:
                    performance[personality]["TN"] += 1
        else:
            number_false += 1
            performance[personalities[int(datas[test_person, -1])]]["FN"] += 1  # FN
            performance[personalities[int(most)]]["FP"] += 1  # FP

# In this function, we are testing for each value of k and each fold. Then we calculate the "Accuracy, Precision and Recall" ratios for each.
def loop():
    for k in [1, 3, 5, 7, 9]:
        accuracy, precision, recall = 0, 0, 0
        for i in intervals:
            k_fold_cross_validation(k, i)
        for personality in performance:
            accuracy += performance[personality]["TP"] + performance[personality]["TN"] / (performance[personality]["TP"] + performance[personality]["FP"]) + performance[personality]["TN"] + performance[personality]["FN"]
            precision += performance[personality]["TP"] / (performance[personality]["TP"] + performance[personality]["FP"])
            recall += performance[personality]["TP"] / (performance[personality]["TP"] + performance[personality]["FN"])
        print(f"    for k = {k} \t Accuracy = ½{100 * accuracy / 16} Precision = ½{100 * precision / 16}\ | Recall = ½{100 * recall / 16}")
    print(f"\n  Total Accuracy = ½{100 * number_true / (number_true + number_false)}")

# We run our test first without normalization and then with normalization.
print("Without Normalization;")
datas = np.array([df_without_normalization.loc[data] for data in range(59999)])
loop()

number_true, number_false = 0, 0
print("\n\n\nWith Normalization;")
datas = np.array([df.loc[data] for data in range(59999)])
loop()