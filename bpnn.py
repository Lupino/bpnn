# Back-Propagation Neural Networks

import math
import random
#import string
try:
    import cPickle as pickle
except:
    import pickle

random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

class Unit:
    def __init__(self, length):
        self.weight = [rand(-0.2, 0.2) for i in range(length)]
        self.change = [0.0] * length
        self.threshold = rand(-0.2, 0.2)
        #self.change_threshold = 0.0
    def calc(self, sample):
        self.sample = sample[:]
        partsum = sum([i * j for i, j in zip(self.sample, self.weight)]) - self.threshold
        self.output = sigmoid(partsum)
        return self.output
    def update(self, diff, rate=0.5, factor=0.1):
        change = [rate * x * diff + factor * c for x, c in zip(self.sample, self.change)]
        self.weight = [w + c for w, c in zip(self.weight, change)]
        self.change = [x * diff for x in self.sample]
        #self.threshold = rateN * factor + rateM * self.change_threshold + self.threshold
        #self.change_threshold = factor
    def get_weight(self):
        return self.weight[:]
    def set_weight(self, weight):
        self.weight = weight[:]


class Layer:
    def __init__(self, input_length, output_length):
        self.units = [Unit(input_length) for i in range(output_length)]
        self.output = [0.0] * output_length
        self.ilen = input_length
    def calc(self, sample):
        self.output = [unit.calc(sample) for unit in self.units]
        return self.output[:]
    def update(self, diffs, rate=0.5, factor=0.1):
        for diff, unit in zip(diffs, self.units):
            unit.update(diff, rate, factor)
    def get_error(self, deltas):
        def _error(deltas, j):
            return sum([delta * unit.weight[j] for delta, unit in zip(deltas, self.units)])
        return [_error(deltas, j) for j  in range(self.ilen)]
    def get_weights(self):
        weights = {}
        for key, unit in enumerate(self.units):
            weights[key] = unit.get_weight()
        return weights
    def set_weights(self, weights):
        for key, unit in enumerate(self.units):
            unit.set_weight(weights[key])



class BPNNet:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no
        self.hlayer = Layer(self.ni, self.nh)
        self.olayer = Layer(self.nh, self.no)

    def calc(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        self.ai = inputs[:] + [1.0]

        # hidden activations
        self.ah = self.hlayer.calc(self.ai)
        # output activations
        self.ao = self.olayer.calc(self.ah)


        return self.ao[:]


    def update(self, targets, rate, factor):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [dsigmoid(ao) * (target - ao) for target, ao in zip(targets, self.ao)]

        # calculate error terms for hidden
        hidden_deltas = [dsigmoid(ah) * error for ah, error in zip(self.ah, self.olayer.get_error(output_deltas))]

        # update output weights
        self.olayer.update(output_deltas, rate, factor)

        # update input weights
        self.hlayer.update(hidden_deltas, rate, factor)
        # calculate error
        return sum([0.5 * (t-o)**2 for t, o in zip(targets, self.ao)])


    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.calc(p[0]))

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.calc(inputs)
                error = error + self.update(targets, N, M)
            if i % 100 == 0:
                print('error %-.10f' % error)
            #self.save_weights('tmp.weights')
    def save_weights(self, fn):
        weights = {
                "olayer":self.olayer.get_weights(),
                "hlayer":self.hlayer.get_weights()
                }
        with open(fn, "wb") as f:
            pickle.dump(weights, f)
    def load_weights(self, fn):
            with open(fn, "rb") as f:
                weights = pickle.load(f)
                self.olayer.set_weights(weights["olayer"])
                self.hlayer.set_weights(weights["hlayer"])

def demo():
    # Teach network XOR function
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]

    # create a network with two input, two hidden, and one output nodes
    n = BPNNet(2, 2, 1)
    # train it with some patterns
    n.train(pat)
    # test it
    n.save_weights("demo.weights")

    n.test(pat)



if __name__ == '__main__':
    demo()
