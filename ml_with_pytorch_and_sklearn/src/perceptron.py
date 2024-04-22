from dataclasses import dataclass
import numpy as np

@dataclass
class Perceptron:
  eta = 0.01 # learning rate, η
  n_iter = 50 # number of iterations
  random_state = 1 # random seed

  # Other attributes
  # w_: np.array
  # b_: np.array
  # errors_: list

  def fit(self, X, Y):
    # X: training vectors, shape = [n_samples, n_features]
    # Y: target values, shape = [n_samples]

    # returns self (Perceptron)

    rgen = np.random.RandomState(self.random_state)
    self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
    self.b_ = np.float_(0)
    self.errors_ = []

    for _ in range(self.n_iter):
      errors = 0
      for xi, target in zip(X, Y):
        update = self.eta * (target - self.predict(xi))
        self.w_ += update * xi
        self.b_ += update
        errors += int(update != 0.0)
      self.errors_.append(errors)

    return self
  

  def net_input(self, X):
      # (x1*w1 + x2*w2 + ... + xn*wn) + b
      return np.dot(X, self.w_) + self.b_
  
  def predict(self, X):
      # σ (x1*w1 + x2*w2 + ... + xn*wn + b) => 1 if >= 0, 0 otherwise
      return np.where(self.net_input(X) >= 0.0, 1, 0)