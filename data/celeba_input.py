import pickle
import random

'''
Function: Creates an Object, which is able to create Batches
 
'''

class CelebA():

	def __init__(self, filename="celeba_pickle.dat"):
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
			self.whole_Data = random.shuffle(self.whole_Data)
			self.current = self.whole_Data

			#take the missing samples:
			batch.append(self.current[:missing])

			#delete those you took out from "current"
			self.current = self.current[missing:]

		#In case our "current" data is big enough
		else:
			#take your batch from "current"
			batch = current[:batch_size]
			#delete those you took out from "current"
			self.current = self.current[batch_size:]

		return batch

if __name__=="__main__":
	pass
