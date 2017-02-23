import pickle
import random
import os
import numpy as np  # just for now


old = True
file_path = ('/'.join(os.path.realpath(__file__).split("/")[:-2]))

'''
Function: Creates an Object, which is able to create Batches
 
'''

def get_data(filename):
	with open(filename, "rb") as f:
		data = pickle.load(f)
	return data

class CelebA():

	def __init__(self, filename= file_path + "/data/pickle/celeba_pickle.dat"):

		# Save Whole Data Set in "whole_Data"
		self.filename = filename
		# Take random sample out of "current", until it is to small to take a batch
		# and then fill it up using "whole_Data"with
		self.current = get_data(self.filename)


	def get_batch(self, batch_size):

		#In case we do not have enough samples in our "current" data
		if len(self.current) < batch_size:

			#missing equals the amount those samples, which we a missing
			missing = batch_size-len(self.current)

			#take all samples that are left
			batch = self.current

			#shuffle "whole_Data" and assign it to the "current" data	
			self.current = get_data(self.filename)
			random.shuffle(self.current)

			#take the missing samples:
			batch += self.current[:missing]

			#delete those you took out from "current"
			self.current = self.current[missing:]

		#In case our "current" data is big enough
		else:
			#take your batch from "current"
			batch = self.current[:batch_size]

			#delete those you took out from "current"
			self.current = self.current[batch_size:]
		
		if old:
			for i in range(len(batch)):
				batch[i] = np.expand_dims(batch[i],-1)
				batch[i] = (batch[i]/255.0 - 0.5) * 2.0
		return batch