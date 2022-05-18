import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import random

seed = 68
random.seed(seed)
import sys

sys.path.append('/home/samuelchen/pkg/code/')


class Loop:
    # Strategies for Sequence Length Imbalance -> Repeat With Gap
    def __init__(self, data):
        self.sequences = data
        self.num_sequences = len(self.sequences)
        self.seq_length = [len(i) for i in self.sequences]
        self.seq_length.sort()
        self.pad_length = self.seq_length[int(self.num_sequences * 0.95)]

    def pad_self_repeats(self, dtype='int32', split='post', value=0.):
        if not hasattr(self.sequences, '__len__'):
            raise ValueError("sequences must be iterable")
        sample_shape = 0
        for s in self.sequences:
            if len(s) > 0:
                sample_shape = np.asarray(s).shape[1:]
                break
        x = (np.ones((self.num_sequences, self.pad_length) + sample_shape) * value).astype(dtype)

        for idx, s in enumerate(self.sequences):
            if not len(s):
                continue
            elif split == 'post':
                tmp = s[:self.pad_length]
            elif split == 'pre':
                tmp = s[-self.pad_length:]

            tmp = np.asarray(tmp, dtype=dtype)

            repeat_seq = np.array([], dtype=dtype)
            while len(repeat_seq) < self.pad_length:
                spacer_length = random.randint(1, 50)
                spacer = [value for _ in range(spacer_length)]
                repeat_seq = np.append(repeat_seq, spacer)
                repeat_seq = np.append(repeat_seq, tmp)

            x[idx, :] = repeat_seq[-self.pad_length:]

        return x

    def run(self):
        return self.pad_self_repeats(), self.pad_length


class Sliding:
    # Strategies for Sequence Length Imbalance -> ASW
    def __init__(self, data):
        self.sequences = data
        self.num_sequences = len(self.sequences)
        self.seq_length = [len(i) for i in self.sequences]

    def straight_output(self, dtype='int32', value=0.):
        if not hasattr(self.sequences, '__len__'):
            raise ValueError("sequences must be iterable")
        sample_shape = 0
        for s in self.sequences:
            if len(s) > 0:
                sample_shape = np.asarray(s).shape[1:]
                break
        x = (np.ones((self.num_sequences, max(self.seq_length)) + sample_shape) * value).astype(dtype)
        for idx, s in enumerate(self.sequences):
            s = np.asarray(s)
            x[idx, :len(s)] = s

        return x

    def run(self):
        return self.straight_output(), self.seq_length


class DealSequence:
    def __init__(self, mode, data, y_encoder, is_x):
        self.data = data
        self.mode = mode
        if mode == 1:
            self.mode = 'loop'
        elif mode == 2:
            self.mode = 'sliding'
        elif mode == 3:
            self.mode = 'no_deal'
        else:
            raise ValueError("choose right mode")

        self.base = 'ATCGN-'
        self.encoder = LabelEncoder()
        self.y_encoder = y_encoder
        self.is_x = is_x

    def deal_sequence(self): # encoding bases to one-hot code
        out = []
        self.encoder.fit(list(self.base))
        if type(self.data) == str:
            out.append(self.encoder.transform(list(x)))
        else:
            for i in self.data:
                out.append(self.encoder.transform(list(i)))

        if self.mode == 'loop':
            out, seq_length = Loop(out).run()
        else:
            out, seq_length = Sliding(out).run()

        return np.array(to_categorical(out, num_classes=len(self.base)), dtype=np.float), seq_length

    def deal_label(self): # encoding label of sequences to one-hot code
        y = self.data
        self.encoder.fit(y)
        if self.y_encoder:
            if np.array(self.encoder.classes_ != self.y_encoder.classes_).all():
                warning(f"Warning not same classes in training and test set")
            useable_classes = set(self.encoder.classes_).intersection(self.y_encoder.classes_)  # 将X和Y放在一起
            try:
                assert np.array(self.encoder.classes_ == self.y_encoder.classes_).all()
            except AssertionError:
                warning(
                    f"not all test classes in training data, only {useable_classes} predictable "
                    f"from {len(self.encoder.classes_)} different classes\ntest set will be filtered so only predictable"
                    f" classes are included")

            try:
                assert len(useable_classes) == len(self.encoder.classes_)  # 判断X和Y的类别长度是否相等
            except AssertionError:
                print("error")
            if not len(useable_classes) == len(self.encoder.classes_):
                global X_test, Y_test
                arr = np.zeros(X_test.shape[0], dtype=int)
                for i in useable_classes:
                    arr[y == i] = 1

                X_test = X_test[arr == 1, :]
                y = y[arr == 1]
                encoded_Y = self.y_encoder.transform(y)
            else:
                encoded_Y = self.encoder.transform(y)

            return to_categorical(encoded_Y, num_classes=len(self.y_encoder.classes_)), self.encoder

        else:
            encoded_Y = self.encoder.transform(y)
            return to_categorical(encoded_Y), self.encoder

    def run(self):
        if self.is_x:
            return self.deal_sequence()
        else:
            return self.deal_label()


class Split:
    def __init__(self, mode, x, y, seq_length):
        self.mode = mode
        self.seq_length = seq_length
        self.x = x
        self.y = y
        self.subseq_length = 250  # leangth of subsequence
        self.n = 35 # if the shortest sequence is smaller than subsequence, then confirm the shortest one can be spilt into this count
                    # subsequences after self-expand

    def change_subseq_len(self, value):
        self.subseq_length = value

    def static_split(self, fit):  # split strategy of Repeat With Gap
        if fit:
            for i in range(4, 400):
                if not self.seq_length % i:
                    self.subseq_length = i
                    break

        batch_size = self.seq_length // self.subseq_length
        print("@@@@@@@@@@@@@@@@@ data_set shape = ", batch_size)
        features = self.x.shape[-1]
        newSeqlength = batch_size * self.subseq_length

        bigarray = []
        for sample in tqdm(self.x):
            sample = np.array(sample[0:newSeqlength], dtype=np.bool)
            subarray = sample.reshape((batch_size, self.subseq_length, features))
            bigarray.append(subarray)
        bigarray = np.array(bigarray)
        x = bigarray.reshape((bigarray.shape[0] * bigarray.shape[1], bigarray.shape[2], bigarray.shape[3]))

        y = []
        for i in self.y:
            y.append(batch_size * [i])
        y = np.array(y)
        if len(y.shape) == 2:
            y = y.flatten()
        elif len(y.shape) == 3:
            y = y.reshape((y.shape[0] * y.shape[1], y.shape[2]))

        return x, y, batch_size, [batch_size] * len(self.x)

    def sliding_window(self): # split strategy of ASW
        features = self.x.shape[-1]
        min_seqLen = min(self.seq_length)
        print("$$$$$$$$$$$$ min seqlength = ", min_seqLen)
        if min_seqLen - self.subseq_length < self.n:
            tmp = self.seq_length.copy()
            tmp.sort()
            for i in tmp:
                if i - self.subseq_length >= self.n:
                    min_seqLen = i
                    break
            print('subseq per seq is: ', self.n)
        else:
            self.n = min_seqLen - self.subseq_length
            print('subseq per seq was changed to: ', self.n)

        bigarray = np.ones((len(self.x), self.n, self.subseq_length, features)).astype(np.bool)
        for index, seq in tqdm(enumerate(self.x)):
            while self.seq_length[index] < min_seqLen:
                self.seq_length[index] *= 2
                self.x[index] *= 2

            step = (self.seq_length[index] - self.subseq_length) // self.n
            for i in range(self.n - 1):
                bigarray[index, i, :] = seq[i * step:self.subseq_length + i * step][:]
            bigarray[index, -1, :] = seq[-self.subseq_length:]

        x = bigarray.reshape((bigarray.shape[0] * bigarray.shape[1], bigarray.shape[2], bigarray.shape[3]))

        y = []
        for i in self.y:
            y.append(self.n * [i])
        y = np.array(y)
        y = y.reshape((y.shape[0] * y.shape[1], y.shape[2]))
        print(x.shape)
        print(y.shape)

        return x, y, self.n, [self.n] * len(self.x)

    def no_deal(self):   # split strategy of Fixed Cut
        features = self.x.shape[-1]
        max_seqLen = max(self.seq_length)
        seqs = []
        count = 0
        max_seqs = max_seqLen // self.subseq_length
        bigarray = np.ones((len(self.x) * max_seqs, self.subseq_length, features)).astype(np.bool)
        print(bigarray.shape)

        for index, seq in tqdm(enumerate(self.x)):
            if self.seq_length[index] < self.subseq_length:
                bigarray[count, :self.seq_length[index]] = seq[:self.seq_length[index]]
                count += 1
                seqs.append(1)
            else:
                nums = self.seq_length[index] // self.subseq_length
                seqs.append(nums)
                for i in range(nums):
                    bigarray[count, :] = seq[i * self.subseq_length: (i+1) * self.subseq_length]
                    count += 1
        x = bigarray[:count, :]

        y = []
        for idx, i in enumerate(self.y):
            y.extend([i] * seqs[idx])
        y = np.array(y)
        print(x.shape)
        print(y.shape)

        return x, y, max_seqLen, seqs


    def test_gen(self):  # all of test data will be split by this method
        features = self.x.shape[-1]
        max_seqLen = max(self.seq_length)
        seqs = []
        count = 0
        max_seqs = max_seqLen // self.subseq_length
        bigarray = np.ones((len(self.x) * max_seqs, self.subseq_length, features)).astype(np.bool)
        print(bigarray.shape)

        for index, seq in tqdm(enumerate(self.x)):
            if self.seq_length[index] < self.subseq_length:
                while self.seq_length[index] < self.subseq_length:
                    self.seq_length[index] *= 2
                    self.x[index] *= 2
                bigarray[count, :self.subseq_length] = self.x[index][:self.subseq_length, :]
                count += 1
                seqs.append(1)
            else:
                nums = self.seq_length[index] // self.subseq_length
                seqs.append(nums)
                for i in range(nums):
                    bigarray[count, :] = seq[i * self.subseq_length: (i + 1) * self.subseq_length]
                    count += 1
        x = bigarray[:count, :]

        y = []
        for idx, i in enumerate(self.y):
            y.extend([i] * seqs[idx])
        y = np.array(y)
        print(x.shape)
        print(y.shape)

        return x, y, max_seqLen, seqs

    def run(self, *args):
        if self.mode == 1:
            return self.static_split(args[0])
        elif self.mode == 2:
            return self.sliding_window()
        elif self.mode == 3:
            return self.no_deal()
