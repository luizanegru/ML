import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score, classification_report

training_sentences = np.load('/Users/luiza/Desktop/IA/data_lab5/training_sentences.npy',allow_pickle=True)
training_labels = np.load('/Users/luiza/Desktop/IA/data_lab5/training_labels.npy',allow_pickle=True)
test_sentences = np.load('/Users/luiza/Desktop/IA/data_lab5/test_sentences.npy',allow_pickle=True)
test_labels = np.load('/Users/luiza/Desktop/IA/data_lab5/test_labels.npy',allow_pickle=True)

def normalize_data(train_data, test_data, type=None):
    if type == 'standard':
        scaler = preprocessing.StandardScaler()
    elif type == 'min_max':
        scaler = preprocessing.MinMaxScaler()
    elif type == 'l2':
        scaler = preprocessing.Normalizer(norm='l2')
    elif type == 'l1':
        scaler = preprocessing.Normalizer(norm='l1')

    if scaler != 'None':
        scaler.fit(train_data)
        scaler.fit(test_data)
        scaler_train = scaler.transform(train_data)
        scaler_test = scaler.transform(test_data)

    return scaler_train, scaler_test



class Bow:
    def __initialize__(self):
        self.words = []
        self.vocab = {}
        self.vocab_len = 0

    def build_vocab(self, data):
        for doc in data:
            for word in doc:
                if word not in self.vocab:
                    self.vocab[word] = len(self.words)
                    self.words.append(word)

        self.vocab_len = len(self.words)
        self.worsd = np.array(self.words)
        return len(self.words)

    def get_features(self, data):
        result = np.zeros((data.shape[0], len(self.words)))
        for i in range(data.shape[0]):
            doc = data[i]
            for word in doc:
                if word in self.vocab:
                    result[i, self.vocab[word]] = result[i, self.vocab[word]] + 1
        return result

object = Bow()
object.build_vocab(training_sentences)

training_features = object.get_features(training_sentences)
testing_features = object.get_features(test_sentences)

print(training_features[0])
print(testing_features[0])

def plot_coefficients(classifier, feature_names, top_features=10):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    print("Top Positive: \n")
    for i in range(10):
        print(feature_names[top_coefficients[i]])
    print("Top Negative:\n")
    for i in range(10,20):
        print(feature_names[top_coefficients[i]])

norm_data = normalize_data(training_features,testing_features,'l2')

obj = svm.SVC(1,kernel = 'linear')
obj.fit(norm_data[0],training_labels)

predictions = obj.predict(norm_data[1])

accuracy = accuracy_score(test_labels,predictions)
print("Accuracy: ")
print(accuracy)
f1_scor = f1_score(test_labels,predictions,average =None)
print("f1_score: ")
print(f1_scor)

plot_coefficients(obj, norm_data[1])
