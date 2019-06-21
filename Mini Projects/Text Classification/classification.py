!pip3 install nltk
import nltk
nltk.download()
import numpy as np
import re
import pickle
from sklearn.datasets import load_files

reviews = load_files("t")