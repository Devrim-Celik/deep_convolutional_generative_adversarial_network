import pickle
import random
import os
import numpy as np  # just for now


old = False
file_path = ('/'.join(os.path.realpath(__file__).split("/")[:-2]))

'''
Function: Creates an Object, which is able to create Batches
 
'''

class CelebA():

	def __init__(self, filename= file_path + "/data/pickle/celeba_pickle.dat"):
		with open(filename, "rb") as f:
			data = pickle.load(f)
		# Save Whole Data Set in "whole_Data"
		self.whole_Data = data
		# Take random sample out of "current", until it is to small to take a batch
		# and then fill it up using "whole_Data"
		self.current = data


	def get_batch(self, batch_size):

		#In case we do not have enough samples in our "current" data
		if len(self.current) < batch_size:

			#missing equals the amount those samples, which we a missing
			missing = batch_size-len(self.current)

			#take all samples that are left
			batch = self.current

			#shuffle "whole_Data" and assign it to the "current" data	
			random.shuffle(self.whole_Data)
			self.current = self.whole_Data

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
			print('Temporary Dimension adding...')
			for i in range(len(batch)):
				batch[i] = np.expand_dims(batch[i],-1)
			print('Temporary Dimension adding finished!')
		return batch