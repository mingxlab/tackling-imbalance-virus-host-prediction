import time
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import sys
import warnings
from logging import warning
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append('/home/samuelchen/pkg/code/')
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, auc, roc_curve
import scipy.stats
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from dataDeal.modules import *
import pickle
import click


@click.command(short_help="train a model on your training files with Transformer")
@click.option('--input_path', '-i', required=True,
              help='path to the dir with training files, generated with make_dataset')
@click.option('--output_path', '-o', default='.', help='path where to save the output')
@click.option('--module', '-m', help='choose data deal module for training')
@click.option('--epochs', '-e', default=int(100), help='maximum number of epochs used for training the model')
@click.option('--architecture', '-a', default=int(1), help='select network architecture')
@click.option('--sample', '-S', default='u', help='sliding_setp')
@click.option('--subseqlength', '-l', default=int(250), help='subseqlength')
@click.option('--step', '-s', default=int(200), help='sliding_setp')
def main(input_path, output_path, module, epochs, architecture, subseqlength, step, sample):
    files = os.listdir(input_path)
    assert "Y_train.csv" in files, f"{input_path} must contain Y_train.csv file, but no such file in {files}"

    if architecture == 1:
        choose_model = 'CNN'
    elif architecture == 2:
        choose_model = 'Bi-LSTM'
    else:
        choose_model = None

    hidden_node = 150
    batch_size_expend = 12
    dropout = 0.2

    model_main(inpath=input_path, outpath=output_path, choose_model=choose_model, datadeal=int(module), step=step, epochs=epochs,
               subseqlength=subseqlength, hidden_node=hidden_node, dropout=dropout, batch_size_expend=batch_size_expend, sample=sample)


class TimeHistory(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        if not hasattr(self, 'times'):
            self.times = []
            self.time_train_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        logs = logs or {}
        self.times.append(int(time.time()) - int(self.time_train_start))


prediction_val = []


class AccuracyHistory(tf.keras.callbacks.Callback):

    def on_train_begin(self):
        if not hasattr(self, 'meanVote_val'):
            self.meanVote_val = []
            self.normalVote_val = []

    def on_epoch_begin(self):
        global prediction_val
        prediction_val = []

    def on_epoch_end(self):
        global prediction_val

        if not len(prediction_val):
            prediction_val = self.model.predict(X_val)

        self.prediction_val = prediction_val

        y_true_small, y_pred_mean_val, y_pred_voted_val, y_pred, y_pred_mean_exact = \
            calc_predictions(X_val, Y_val, do_print=False, y_pred=self.prediction_val)
        self.normalVote_val.append(metrics.accuracy_score(y_true_small, y_pred_voted_val))
        self.meanVote_val.append(metrics.accuracy_score(y_true_small, y_pred_mean_val))

class PredictionHistory(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        p = np.random.permutation(len(Y_val))
        shuffled_X = X_val[p]
        shuffled_Y = Y_val[p]
        self.predhis = (self.model.predict(shuffled_X[0:10]))
        y_pred = np.argmax(self.predhis, axis=-1)
        y_true = np.argmax(shuffled_Y, axis=-1)[0:10]
        print(f"Predicted: {y_pred}")
        print(f"True:      {y_true}")
        table = pd.crosstab(
            pd.Series(y_true),
            pd.Series(y_pred),
            rownames=['True'],
            colnames=['Predicted'],
            margins=True)
        print(table)

class History(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        if not hasattr(self, 'epoch'):
            self.epoch = []
            self.history = {}

    def on_epoch_end(self, epoch, logs={}):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


def deal_data_into_sequences(directory, step, subseqlength, mode):
    print("#################################################")
    print("read dataset file and deal it into sequences")
    print("#################################################")

    Y_train_old = pd.read_csv(directory + '/Y_train.csv', delimiter='\t', dtype='str', header=None)[1].values
    Y_test_old = pd.read_csv(directory + '/Y_test.csv', delimiter='\t', dtype='str', header=None)[1].values
    X_train_old = pd.read_csv(directory + '/X_train.csv', delimiter='\t', dtype='str', header=None)[1].values
    X_test_old = pd.read_csv(directory + '/X_test.csv', delimiter='\t', dtype='str', header=None)[1].values
    Y_val_old = pd.read_csv(directory + '/Y_val.csv', delimiter='\t', dtype='str', header=None)[1].values
    X_val_old = pd.read_csv(directory + '/X_val.csv', delimiter='\t', dtype='str', header=None)[1].values


    global X_test, X_train, X_val, Y_test, Y_train, Y_val

    X_train, train_seq_length = DealSequence(mode, X_train_old, None, True).run()
    X_val, val_seq_length = DealSequence(mode, X_val_old, None, True).run()
    X_test, test_seq_length = DealSequence(2, X_test_old, None, True).run()

    Y_train, y_encoder = DealSequence(mode, Y_train_old, None, False).run()
    Y_val, _ = DealSequence(mode, Y_val_old, y_encoder, False).run()
    Y_test, _ = DealSequence(2, Y_test_old, y_encoder, False).run()


    return train_seq_length, val_seq_length, test_seq_length


def models(inpath, outpath, choose_model, nodes, epochs, dropout, batch_size_expend, sample):

    global X_train, X_test, Y_train, model
    timesteps = X_test.shape[1]

    batchsize = 64
    batchsize = batchsize * max(1, batch_size_expend)

    if choose_model == 'CNN':
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv1D(nodes, 9, input_shape=(timesteps, X_test.shape[-1])))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
        model.add(tf.keras.layers.MaxPooling1D(3))
        model.add(tf.keras.layers.Conv1D(nodes, 9))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
        model.add(tf.keras.layers.MaxPooling1D(3))
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(nodes, return_sequences=True, recurrent_activation="sigmoid"),
            input_shape=(timesteps, X_test.shape[-1])))

        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nodes, recurrent_activation="sigmoid")))

        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.Dense(nodes, activation='elu'))
        model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.Dense(Y_train.shape[-1], activation='softmax'))

    elif choose_model == 'Bi-LSTM':
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nodes, return_sequences=True, dropout=dropout),
                                                input_shape=(timesteps, X_test.shape[-1])))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nodes, return_sequences=True, dropout=dropout)))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nodes, dropout=dropout)))
        model.add(tf.keras.layers.Dense(nodes, activation='elu'))
        model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.Dense(Y_train.shape[-1], activation='softmax'))


    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=200, verbose=1,
                                                      restore_best_weights=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'], sample_weight_mode=None)

    filepath = outpath + "/best_acc_model.hdf5"
    filepath2 = outpath + "/best_loss_model.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                                    mode='max')
    checkpoint2 = tf.keras.callbacks.ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True,

    predictions = PredictionHistory()
    time_callback = TimeHistory()
    callbacks_list = [checkpoint, checkpoint2, predictions, time_callback]
    print(X_train.shape)
    clw = None
    class_weight1 = {4: 101.78, 0: 87.74, 10: 79.52, 3: 74.84, 12: 73.75, 7: 59.17, 5: 38.55, 15: 34.86, 14: 23.89,
                     8: 22.42, 6: 11.89, 9: 11.41, 13: 9.67, 11: 7.95, 16: 7.82, 1: 2.26, 2: 1.00}
    class_weight2 = {4: 508.90, 0: 508.90, 10: 508.90, 3: 508.90, 12: 73.75, 7: 59.17, 5: 38.55, 15: 34.86, 14: 23.89,
                     8: 22.42, 6: 11.89, 9: 11.41, 13: 9.67, 11: 7.95, 16: 7.82, 1: 2.26, 2: 1.00}
    class_weight3 = {4: 169.63, 0: 169.63, 10: 169.63, 3: 169.63, 12: 73.75, 7: 59.17, 5: 38.55, 15: 34.86, 14: 23.89,
                     8: 22.42, 6: 11.89, 9: 11.41, 13: 9.67, 11: 7.95, 16: 7.82, 1: 2.26, 2: 1.00}
    class_weight4 = {4: 254.45, 0: 254.45, 10: 254.45, 3: 254.45, 12: 73.75, 7: 59.17, 5: 38.55, 15: 34.86, 14: 23.89,
                     8: 22.42, 6: 11.89, 9: 11.41, 13: 9.67, 11: 7.95, 16: 7.82, 1: 2.26, 2: 1.00}
    class_weight5 = {4: 1017.8, 0: 1017.8, 10: 1017.8, 3: 1017.8, 12: 73.75, 7: 59.17, 5: 38.55, 15: 34.86, 14: 23.89,
                     8: 22.42, 6: 11.89, 9: 11.41, 13: 9.67, 11: 7.95, 16: 7.82, 1: 2.26, 2: 1.00}
    class_weight6 = {4: 1696.33, 0: 1696.33, 10: 1696.33, 3: 1696.33, 12: 73.75, 7: 59.17, 5: 38.55, 15: 34.86, 14: 23.89,
                     8: 22.42, 6: 11.89, 9: 11.41, 13: 9.67, 11: 7.95, 16: 7.82, 1: 2.26, 2: 1.00}
    if sample == '50':
        clw = class_weight1
    elif sample == '10':
        clw = class_weight2
    elif sample == '30':
        clw = class_weight3
    elif sample == '20':
        clw = class_weight4
    elif sample == '05':
        clw = class_weight5
    elif sample == '03':
        clw = class_weight6


    hist = model.fit(X_train, Y_train, epochs=epochs, callbacks=callbacks_list, batch_size=batchsize,
                     validation_data=(X_val, Y_val), class_weight=clw,
                     shuffle=True)
    times = time_callback.times


    if not os.path.isfile(outpath + "/history.csv"):
        histDataframe = pd.DataFrame(hist.history)
        cols = ["acc", "loss", "val_acc", "val_loss"]
        histDataframe = histDataframe[cols]
        histDataframe = histDataframe.assign(time=times)
        histDataframe.to_csv(outpath + "/history.csv")
    else:
        histDataframe = pd.DataFrame(hist.history)
        histDataframe = histDataframe.assign(time=times)
        histDataframe.to_csv(outpath + "/history.csv", mode='a', header=False)

    return


def calc_predictions(name, Y, test_samples, y_pred, outpath, do_print=False):
    classes = {0: 'Artibeus lituratus', 1: 'Bos taurus', 2: 'Canis lupus', 3: 'Capra hircus', 4: 'Cerdocyon thous',
               5: 'Desmodus rotundus', 6: 'Eptesicus fuscus', 7: 'Equus caballus', 8: 'Felis catus',
               9: 'Homo sapiens', 10: 'Lasiurus borealis', 11: 'Mephitis mephitis', 12: 'Nyctereutes procyonoides',
               13: 'Procyon lotor', 14: 'Tadarida brasiliensis', 15: 'Vulpes lagopus', 16: 'Vulpes vulpes'}

    classes_count = len(classes)

    def standard_probability(y_true, y_pred, label):
        is_true = []
        probability = []
        for idx, i in enumerate(y_pred):
            if np.argmax(y_true[idx]) != label:
                is_true.append(0)
            else:
                is_true.append(1)
            probability.append(i[label])
        df = pd.DataFrame({'True label': is_true, 'Probability': probability})
        df.to_csv(outpath + name + str(label) + ".csv")

    def print_predictions(y_true, y_pred, y_true_small, y_pred_voted, y_pred_sum, y_pred_mean_weight_std,
                          y_pred_mean_weight_ent):

        acc = {}
        table = pd.crosstab(
            pd.Series(y_encoder.inverse_transform(y_true)),
            pd.Series(y_encoder.inverse_transform(y_pred)),
            rownames=['True'],
            colnames=['Predicted'],
            margins=True)
        print("standard version")
        table.to_csv(outpath + '/confusion_matrix.csv')
        print(table.to_string())
        accuracy = metrics.accuracy_score(y_true, y_pred) * 100
        print("standard version")
        print("acc = " + str(accuracy))
        acc["standard version"] = accuracy

        table = pd.crosstab(
            pd.Series(y_encoder.inverse_transform(y_true_small)),
            pd.Series(y_encoder.inverse_transform(y_pred_voted)),
            rownames=['True'],
            colnames=['Predicted'],
            margins=True)
        print("vote version")
        print(table.to_string())
        accuracy = metrics.accuracy_score(y_true_small, y_pred_voted) * 100
        print("vote version")
        print("acc = " + str(accuracy))
        acc["vote version"] = accuracy

        table = pd.crosstab(
            pd.Series(y_encoder.inverse_transform(y_true_small)),
            pd.Series(y_encoder.inverse_transform(y_pred_sum)),
            rownames=['True'],
            colnames=['Predicted'],
            margins=True)
        print("mean version")
        print(table.to_string())
        accuracy = metrics.accuracy_score(y_true_small, y_pred_sum) * 100
        print("mean version")
        print("acc = " + str(accuracy))
        acc["mean"] = accuracy


        table = pd.crosstab(
            pd.Series(y_encoder.inverse_transform(y_true_small)),
            pd.Series(y_encoder.inverse_transform(y_pred_mean_weight_ent)),
            rownames=['True'],
            colnames=['Predicted'],
            margins=True)
        print("entropie version")
        print(table.to_string())
        accuracy = metrics.accuracy_score(y_true_small, y_pred_mean_weight_ent) * 100
        print("entropie version")
        print("acc = " + str(accuracy))
        acc["entropie version"] = accuracy

        table = pd.crosstab(
            pd.Series(y_encoder.inverse_transform(y_true_small)),
            pd.Series(y_encoder.inverse_transform(y_pred_sum)),
            rownames=['True'],
            colnames=['Predicted'],
            margins=True)
        print("std version")
        print(table.to_string())
        accuracy = metrics.accuracy_score(y_true_small, y_pred_mean_weight_std) * 100
        print("std-div version")
        print("acc = " + str(accuracy))
        acc["std version"] = accuracy

        seq_level = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
        seq_level.to_csv(outpath + '/' + name + '_y_sequence_level.csv', index=False)
        subseq_level = pd.DataFrame({'y_true_small': y_true_small, 'y_pred_vote': y_pred_voted,
                                     'y_pred_mean': y_pred_sum, 'y_pred_std': y_pred_mean_weight_std})
        subseq_level.to_csv(outpath + '/' + name + '_y_subsequence_level.csv', index=False)
        overall_acc = pd.DataFrame(acc.items())
        overall_acc.to_csv(outpath + "/overall_acc.csv")

    y_pred_mean = []
    y_pred_mean_exact = []
    weigth_entropy = []
    y_pred_mean_weight_ent = []
    weigth_std = []
    y_pred_mean_weight_std = []

    standard_probability(Y, y_pred, 4)
    standard_probability(Y, y_pred, 2)

    for i in y_pred:
        weigth_std.append(np.std(i))
        number_classes = Y.shape[-1]
        weigth_entropy.append(scipy.stats.entropy(scipy.stats.norm.pdf(i, loc=1 / number_classes, scale=0.5)))

    test_pred_counts = 0
    for i in test_samples:
        sample_pred_mean = np.array(
            np.sum(y_pred[test_pred_counts:i + test_pred_counts],
                   axis=0) / i)
        y_pred_mean.append(np.argmax(sample_pred_mean))
        y_pred_mean_exact.append(sample_pred_mean)

        sample_weigths = weigth_entropy[
                         test_pred_counts:i + test_pred_counts]
        sw_normalized = np.array(sample_weigths / np.sum(sample_weigths)).reshape(-1, 1)
        y_pred_mean_weight_ent.append(np.argmax(np.array(
            np.sum(
                np.array(
                    y_pred[test_pred_counts:i + test_pred_counts]) * sw_normalized,
                axis=0) / i)))

        sample_weigths = weigth_std[test_pred_counts:i + test_pred_counts]
        sw_normalized = np.array(sample_weigths / np.sum(sample_weigths)).reshape(-1, 1)
        y_pred_mean_weight_std.append(np.argmax(np.array(
            np.sum(
                np.array(
                    y_pred[test_pred_counts:i + test_pred_counts]) * sw_normalized,
                axis=0) / i)))
        test_pred_counts += i

    y_pred = np.argmax(y_pred, axis=-1)

    y_true = np.argmax(Y, axis=-1)
    y_true_small, y_pred_voted = [], []

    test_pred_count = 0
    for idx in range(min(len(test_samples), len(y_pred))):
        i = test_samples[idx]
        arr = np.bincount(y_pred[test_pred_count:i + test_pred_count])
        best = np.argwhere(arr == np.amax(arr)).flatten()
        y_pred_voted.append(np.random.permutation(best)[
                                0])
        y_true_small.append(np.argmax(
            np.array(np.bincount(y_true[test_pred_count:i + test_pred_count]))))

        test_pred_count += i

    if do_print:
        print_predictions(y_true, y_pred, y_true_small, y_pred_voted, y_pred_mean, y_pred_mean_weight_std,
                          y_pred_mean_weight_ent)


    return y_true_small, y_pred_mean, y_pred_voted, y_pred, np.array(y_pred_mean_exact)


def model_main(inpath, outpath, choose_model, datadeal, step, epochs, subseqlength, hidden_node,
               dropout, batch_size_expend, sample):
    print(datadeal)

    fit = True if choose_model == 1 else None
    global X_train, X_test, X_val, Y_train, Y_test, Y_val, batch_size, SEED, y_encoder, number_subsequences, test_samples
    Y_train_old = pd.read_csv(inpath + '/Y_train.csv', delimiter='\t', dtype='str', header=None)[1].values
    print(type(Y_train_old))
    Y_train, y_encoder = DealSequence(datadeal, Y_train_old, None, False).run()
    print("#################################################")
    print("class to label:")
    print(*(zip(y_encoder.transform(y_encoder.classes_), y_encoder.classes_)))
    print("#################################################")

    X_train_seq_len, X_val_seq_len, X_test_seq_len = deal_data_into_sequences(directory=inpath, step=step,
                                                                                          subseqlength=subseqlength,
                                                                                          mode=datadeal)
    if hidden_node < X_test.shape[-1]:
        warning("use at least as many nodes as number of hosts to predict, better twice as much")

    print("#################################################")
    print("split data into subsequences")
    print("#################################################")
    X_train, Y_train, number_subsequences, _ = Split(datadeal, X_train, Y_train, X_train_seq_len).run(fit)
    X_val, Y_val, number_subsequences, _ = Split(datadeal, X_val, Y_val, X_val_seq_len).run(fit)
    X_test, Y_test, _, test_samples = Split(datadeal, X_test, Y_test, X_test_seq_len).test_gen()

    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    print("#################################################")
    print("train the model")
    print("#################################################")

    models(inpath=inpath, outpath=outpath, choose_model=choose_model, nodes=hidden_node, epochs=epochs,
           dropout=dropout, batch_size_expend=batch_size_expend, sample=sample)

    model_path1 = f"{outpath}/best_loss_model.hdf5"
    model_path2 = f"{outpath}/best_acc_model.hdf5"

    print("#################################################")
    print("test best models")
    print("#################################################")
    for model_path in (model_path1, model_path2):
        print("load model:")
        print(choose_model)
        print(model_path)
        name = model_path[-20:-5]
        model = tf.keras.models.load_model(model_path)
        pred = model.predict(X_test)
        y_true_small, y_pred_mean, y_pred_voted, y_pred, y_pred_mean_exact = calc_predictions(name, Y_test,
                                                                                              test_samples,
                                                                                              y_pred=pred,
                                                                                              outpath=outpath,
                                                                                              do_print=True)
    tf.keras.backend.clear_session()
    del model


X_test = []
X_val = []
X_train = []
Y_test = []
Y_val = []
Y_train = []
test_sample = []
SEED = 42
batch_size = 48
y_encoder = None
directory = ''
number_subsequences = 1
if __name__ == '__main__':
    main()








