import math
import numpy as np

#preprocess sample of data
def preprocess(sam):
    sam = sam.lower()  # lowercase
    # remove punctuation 
    clean_sam = ''.join([char if char.isalnum() or char.isspace() else '' for char in sam])
    return clean_sam.split()  # tokenize

#extract n-grams
def extract_nn(words, n):
    nn = []
    #go through the word list and get the list of n-grams
    for i in range(len(words) - n + 1):
        ngram = '_'.join(words[i:i + n])
        nn.append(ngram)
    return nn

#extract features based on the context
def f_ext(cc):
    features = []
    words = preprocess(cc)
    left_cc = words[:5]
    right_cc = words[6:]
    features.extend(left_cc + right_cc)

    if len(words) > 1:
        if len(left_cc) >= 1:
            features.append(f"word_-1:{left_cc[-1]}")
        if len(right_cc) >= 1:
            features.append(f"word_1:{right_cc[0]}")
        if len(left_cc) >= 2:
            features.append(f"word_-2:{left_cc[-2]}")
        if len(right_cc) >= 2:
            features.append(f"word_2:{right_cc[1]}")

    features.extend(extract_nn(left_cc, 2))
    features.extend(extract_nn(right_cc, 2))
    features.extend(extract_nn(left_cc, 3))
    features.extend(extract_nn(right_cc, 3))

    return features

#log-likelihood
def log_likelihood(aa, bb, tt):
    llh = {}
    for feature, senses in aa.items():
        llh[feature] = {}
        for sense, count in senses.items():
            other_sense_count = tt - bb[sense]
            P_feature_given_sense = (count + 1) / (bb[sense] + 2)
            P_feature_given_other = (senses.get('other', 0) + 1) / (other_sense_count + 2)
            llh[feature][sense] = math.log(P_feature_given_sense / P_feature_given_other)
    return llh

#rank features for decision list based on llh
def rank_decision_list(llh):
    ranked_dl = []
    for feature, senses in llh.items():
        best_sense = max(senses, key=senses.get)
        best_score = senses[best_sense]
        ranked_dl.append((feature, best_sense, best_score))
    ranked_dl.sort(key=lambda x: x[2], reverse=True)
    return ranked_dl

#train decision list
def train_decision_list(training_data, word_type):
    aa = {}
    bb = {}
    tt = 0
    for line in training_data:
        if ':' not in line:
            continue
        kk, dd = line.split(':', 1)

        if word_type == 'bass':
            sense = 'fish' if kk.startswith('*bass') else 'music'
        elif word_type == 'sake':
            sense = 'beer' if kk.startswith('*sake') else 'cause'

        feature = f_ext(dd)
        for feature in feature:
            if feature not in aa:
                aa[feature] = {}
            if sense not in aa[feature]:
                aa[feature][sense] = 0
            aa[feature][sense] += 1

        if sense not in bb:
            bb[sense] = 0
        bb[sense] += 1
        tt += 1

    llh = log_likelihood(aa, bb, tt)
    return rank_decision_list(llh), bb

#classify test data
def classify(test_data, ranked_decision_list, word_type):
    predictions = []
    for line in test_data:
        if ':' not in line:
            continue
        _, cc = line.split(':', 1)  

        features = f_ext(cc)
        best_sense = None
        for feature, sense, score in ranked_decision_list:
            if feature in features:
                best_sense = sense
                break
        predictions.append(best_sense)
    return predictions

#baseline prediction
def baseline_predict(test_data, most_frequent_sense):
    return [most_frequent_sense] * len(test_data)

#accuracy of classifier
def evaluate(predictions, rl):
    #number of correct predictions
    correct = sum([1 for pred, actual in zip(predictions, rl) if pred == actual])
    # divided by total predictions
    accuracy = correct / len(rl)
    return accuracy

#confusion matrix 
def cal_cm(rl, predictions):
    z = sorted(set(rl))
    matrix = [[0 for _ in z] for _ in z]
    label_index = {kk: i for i, kk in enumerate(z)}

    for actual, predicted in zip(rl, predictions):
        matrix[label_index[actual]][label_index[predicted]] += 1

    return np.array(matrix)

'''
true_positives = ww
false_positives = vv
false_negatives = uu
'''

#macro and micro precision and recall
def cal_mm(rl, predictions):
    z = sorted(set(rl))
    ww = {kk: 0 for kk in z}
    vv = {kk: 0 for kk in z}
    uu = {kk: 0 for kk in z}

    # Calculate true positives, false positives, and false negatives
    for actual, predicted in zip(rl, predictions):
        if actual == predicted:
            ww[actual] += 1
        else:
            vv[predicted] += 1
            uu[actual] += 1

    #p = precision and r= recall per label
    p = {}
    r = {}

    for kk in z:
        tp = ww[kk]
        fp = vv[kk]
        fn = uu[kk]

        p[kk] = tp / (tp + fp) if (tp + fp) > 0 else 0
        r[kk] = tp / (tp + fn) if (tp + fn) > 0 else 0

    macro_p = sum(p[kk] for kk in z) / len(z)
    macro_r = sum(r[kk] for kk in z) / len(z)

    total_tp = sum(ww.values())
    total_fp = sum(vv.values())
    total_fn = sum(uu.values())

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    return macro_p, micro_p, macro_r, micro_r

def load_data(file_loc):
    with open(file_loc, 'r') as file:  
        data = file.readlines()  
    return data  

def real_lab(test_data, word_type):
    rl = []  
    for line in test_data:  
        if ':' not in line:  
            continue
        kk, _ = line.split(':', 1)  #extract labells

        #word type = bass/ sake 
        if word_type == 'bass':
            actual_label = 'fish' if kk.startswith('*bass') else 'music'  
        elif word_type == 'sake':
            actual_label = 'beer' if kk.startswith('*sake') else 'cause'  

        rl.append(actual_label)  

    return rl  

#file paths - change here the files for training and testing data
bass_training_file = '/bass.trn'
bass_test_file = '/bass.tst'
sake_training_file = '/sake.trn'
sake_test_file = '/sake.tst'

#load training and test files
bass_training_values = load_data(bass_training_file)
bass_test_values = load_data(bass_test_file)
sake_training_values = load_data(sake_training_file)
sake_test_values = load_data(sake_test_file)

#extract actual labes - test data
bass_rl = real_lab(bass_test_values, 'bass')
sake_rl = real_lab(sake_test_values, 'sake')

#train classifier 
bass_ranked_decision_list, bass_sense_counts = train_decision_list(bass_training_values, 'bass')
sake_ranked_decision_list, sake_sense_counts = train_decision_list(sake_training_values, 'sake')

#classify - test data 
bass_predictions = classify(bass_test_values, bass_ranked_decision_list, 'bass')
sake_predictions = classify(sake_test_values, sake_ranked_decision_list, 'sake')

#most frequent sense for bass/sake
bass_most_frequent_sense = max(bass_sense_counts, key=bass_sense_counts.get)  
sake_most_frequent_sense = max(sake_sense_counts, key=sake_sense_counts.get)  
#baseline predictions for bass/sake
bass_baseline_predictions = baseline_predict(bass_test_values, bass_most_frequent_sense)  
sake_baseline_predictions = baseline_predict(sake_test_values, sake_most_frequent_sense)  

#get accuracy and baseline accuracy 
bass_accuracy = evaluate(bass_predictions, bass_rl)
sake_accuracy = evaluate(sake_predictions, sake_rl)
bass_baseline_accuracy = evaluate(bass_baseline_predictions, bass_rl)
sake_baseline_accuracy = evaluate(sake_baseline_predictions, sake_rl)

#calculate confusion matrices 
bass_cm = cal_cm(bass_rl, bass_predictions)
sake_cm = cal_cm(sake_rl, sake_predictions)

#calculate metrics 
bass_me = cal_mm(bass_rl, bass_predictions)
sake_me = cal_mm(sake_rl, sake_predictions)

#calculate % absolute 
#difference bw accuracy and baseline accuracy
bass_ac = (bass_accuracy - bass_baseline_accuracy) * 100
sake_ac = (sake_accuracy - sake_baseline_accuracy) * 100

#calculate % error reduction 
bass_er = ((1 - bass_baseline_accuracy) - (1 - bass_accuracy)) / (1 - bass_baseline_accuracy) * 100
sake_er = ((1 - sake_baseline_accuracy) - (1 - sake_accuracy)) / (1 - sake_baseline_accuracy) * 100

print(f"Bass acc: {bass_accuracy:.2f}")
print(f"Bass baseline acc: {bass_baseline_accuracy:.2f}")
print(f"Bass % absolute change: {bass_ac:.2f}")
print(f"Bass % error reduction: {bass_er:.2f}")
print(f"Bass confusion matrix:\n{bass_cm}")
print(f"Bass metrics: macro precision: {bass_me[0]:.2f}, micro precision: {bass_me[1]:.2f}, macro recall: {bass_me[2]:.2f}, micro recall: {bass_me[3]:.2f}")

print(f"Sake acc: {sake_accuracy:.2f}")
print(f"Sake baseline acc: {sake_baseline_accuracy:.2f}")
print(f"Sake % absolute change: {sake_ac:.2f}")
print(f"Sake % error reduction: {sake_er:.2f}")
print(f"Sake confusion matrix:\n{sake_cm}")
print(f"Sake metrics: macro precision: {sake_me[0]:.2f}, micro precision: {sake_me[1]:.2f}, macro recall: {sake_me[2]:.2f}, micro recall: {sake_me[3]:.2f}")
