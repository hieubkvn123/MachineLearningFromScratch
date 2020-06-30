import numpy as np

# this is naive bayes implementation
class_1 = np.array([[1, 2, 3], [1, 1, 2], [2, 2, 2],
                    [2, 1, 2], [2, 2, 1], [1, 3, 1],
                    [3, 2, 1], [3, 1, 3], [3, 3, 3]])

class_2 = np.array([[4, 2, 4], [4, 5, 3], [4, 4, 4],
                    [5, 6, 7], [5, 5, 5], [4, 5, 4],
                    [5, 6, 5], [6, 5, 6], [6, 6, 6]])

x = np.concatenate((class_1, class_2))
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

x_test = np.array([[1,2,2], [2,1,1], [3,2,2], [4,2,4], [6,3,2], [4,5,3]])
y_test = np.array([1, 1, 1, -1, -1, -1])

class NaiveBayes(object):
    def __init__(self):
        self.x = np.array([])
        self.y = np.array([])

    def get_probability_y(self, y_i):
        ### Get the probability of each class happening ###
        y_i = self.y[np.where(self.y == y_i)]
        N_y_i = y_i.shape[0]

        P_yi = float( N_y_i / self.N )

        return P_yi

    def get_probability_x_given_y(self, x, y_i):
        ### the probability of x happining given y ###
        
        # first, get the training set where y = y_i
        x_train = self.x[np.where(self.y == y_i)]
        train_size = x_train.shape[0]

        occurences = list()

        ### Loop through the testing data ###
        for i in range(x.shape[0]):
            occurence_count = 0
            for j in range(x_train.shape[0]):
                if(x[i] in x_train[j]):
                    occurence_count += 1

            occurences.append(occurence_count)

        ### calculate probability of all x_i ###
        P_xi_given_y = 1
        for occurence in occurences:
            P_xi_given_y *= occurence

        return P_xi_given_y

    def fit(self, x, y):
        if(x.shape[0] == y.shape[0]):
            self.x = x
            self.y = y
            self.N = x.shape[0]
        else:
            print("[INFO] Input and output must have the same length ...")
            
    def predict(self, x, probability=True):
        if(len(x.shape) < 2):
            print("[INFO] Input data must be an array of vectors ... ")
            return None

        if(x.shape[1] != self.x.shape[1]):
            print("[INFO] Input must have the same shape as the training dataset")
            return None

        outputs = list()
        max_probabilities = list()

        for i in range(x.shape[0]):
            probabilities = list()
            for y_i in np.unique(self.y):
                p = self.get_probability_y(y_i) * self.get_probability_x_given_y(x[i], y_i)
                probabilities.append(p)

            class_ = np.unique(self.y)[np.argmax(probabilities)]
            max_probability = probabilities[np.argmax(probabilities)] / sum(probabilities)

            max_probabilities.append(max_probability)
            outputs.append(class_)

        if(probability):
            return outputs, max_probabilities
        else:
            return outputs

def accuracy_score(y_test, y):
    correct = 0
    for i, y_ in enumerate(y_test):
        if(y_ == y[i]):
            correct += 1

    accuracy = float(correct / y.shape[0])

    return "{0:.2f}".format(accuracy)

nb = NaiveBayes()
nb.fit(x,y)
outputs, probabilities = nb.predict(x_test)

for p, y_i in zip(probabilities, outputs):
    print("Predicted class : " + str(y_i) + " | Probability = {0:.2f}".format(p))


accuracy = accuracy_score(outputs, y_test)
print("----------------------------------------------------------")
print("[INFO] Test accuracy : " + str(accuracy))
