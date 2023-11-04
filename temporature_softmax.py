import numpy as np
import matplotlib.pyplot as plt

# Initialize random probability distribution
x = np.random.rand(5)
x = x/np.sum(x)

def softmax(x):
  score = np.exp(x)/np.sum(np.exp(x))
  return score

def softmax_scale(x, temp):
  x = [i/temp for i in x]
  score_scale = softmax(x)
  return score_scale

score_1 = softmax(x)
score_2 = softmax_scale(x, 2)

def _plot_line(score1, score2):
  assert len(score1) == len(score2)
  classes = np.arange(len(score1))
  plt.figure(figsize=(10, 6))
  plt.plot(classes, score1, label="Softmax Score")
  plt.plot(classes, score2, label = "Softmax Temperature Score")
  # plt.ylim([0, 1])
  plt.legend()
  plt.title("Softmax distribution score")
  plt.show()

_plot_line(score_1, score_2)