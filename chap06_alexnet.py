from datetime import datetime
import math
import time
import tensorflow as tf 

batch_size = 32
num_batches = 100

def print_activations(t):
	print(t.op.name, ' ', t.get_shape().as_list())

def inference(images):
	