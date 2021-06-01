from typing import Tuple

import sys
import string
import random
import typing
import unicodedata
import numpy as np
import csv
from math import log
from collections import Counter

random.seed(42)

ALPHABET = list(string.ascii_lowercase) + [" ", "."]
LETTER_TO_IDX = dict(map(reversed, enumerate(ALPHABET)))

## Load the data specifying the unigram and bigram probabilits of English. 

path_to_letter_probabilities = './data/letter_probabilities.csv'
path_to_transition_probabilties = './data/letter_transition_matrix.csv'

with open(path_to_letter_probabilities) as letter_prob_file:
	letter_probabilities = list(csv.reader(letter_prob_file))[0]
letter_probabilities = [float(prob) for prob in letter_probabilities]

with open(path_to_transition_probabilties) as transition_file:
	letter_transition_matrix = list(csv.reader(transition_file))
letter_transition_matrix = [[float(prob) for prob in letter_transition_matrix[i]] for i in range(len(letter_transition_matrix))]

## Assign value to hyper-parameters.
bp_sample_window = 100

class Cipher:
	'''
	Class to represent a general cipher.

	if not `has_breakpoint`, 
		the breakpoint is arbitrarily set to be at the end of the ciphertext,
		and the second substition is arbitrarily chosen.
	'''

	def __init__(self, has_breakpoint: bool, bp: int, cipherbet_1: str, cipherbet_2: str, ciphertext_length: int):
		self.has_breakpoint = has_breakpoint
		self.bp = bp
		self.cipherbet_1 = cipherbet_1
		self.cipherbet_2 = cipherbet_2
		self.ciphertext_length = ciphertext_length

	@classmethod
	def create_initial_cipher(cls, ciphertext: str, has_breakpoint: bool, bp: int):
		'''
		Initialize the cipher based on heuristics derived from the properties of English.
		'''
		cipherbet_1 = initial_guess_heuristic(ciphertext[:bp])
		cipherbet_2 = initial_guess_heuristic(ciphertext[bp:])
		ciphertext_length = len(ciphertext)
		return cls(has_breakpoint, bp, cipherbet_1, cipherbet_2, ciphertext_length)

	def invert(self, ciphertext: str) -> str:
		'''
		Decode ciphertext using the cipher represented in self.
		'''
		CIPHERBET_1_LETTER_TO_ID = dict(map(reversed, enumerate(self.cipherbet_1)))
		plaintext_1 = "".join(ALPHABET[CIPHERBET_1_LETTER_TO_ID[c]] for c in ciphertext[:self.bp])

		if not self.has_breakpoint:
			return plaintext_1

		CIPHERBET_2_LETTER_TO_ID = dict(map(reversed, enumerate(self.cipherbet_2)))
		plaintext_2 = "".join(ALPHABET[CIPHERBET_2_LETTER_TO_ID[c]] for c in ciphertext[self.bp:])

		return plaintext_1 + plaintext_2

	def log_likelihood(self, ciphertext: str) -> float:
		'''
		Compute and return the log likelihood of the cipher represented by self, given the ciphertext.
		'''
		plaintext = self.invert(ciphertext)
		result = 0.0
		for i, p in enumerate(plaintext):
			if i==0:
				probability = letter_probabilities[LETTER_TO_IDX[p]]
				result += log(probability if probability != 0 else 0.000001, 10.0)
			else:
				q = plaintext[i-1]
				probability = letter_transition_matrix[LETTER_TO_IDX[p]][LETTER_TO_IDX[q]]
				result += log(probability if probability != 0 else 0.000001, 10.0)
		return result

	def sample_next_cipher(self, bp_sample_window):
		'''
		Samples a new cipher according to the proposal distribution, described below.
		For a simple cipher without breakpoints,
			samples a new cipher uniformly at random such that it 
			differs from the current cipher at exactly two locations
		For a cipher with breakpoints, each component simple cipher is sampled as above, 
			and a new breakpoint is sampled uniformly from a fixed window around the current breakpoint.
		'''
		i, j = random.sample(range(len(ALPHABET)), 2)
		next_cipherbet_1 = self.cipherbet_1.copy()
		next_cipherbet_1[i], next_cipherbet_1[j] = self.cipherbet_1[j], self.cipherbet_1[i]

		if self.has_breakpoint:
			ii, jj = random.sample(range(len(ALPHABET)), 2)
			next_cipherbet_2 = self.cipherbet_2.copy()
			next_cipherbet_2[i], next_cipherbet_2[j] = self.cipherbet_2[j], self.cipherbet_2[i]

			low = max(0, self.bp - bp_sample_window)
			high = min(self.bp + bp_sample_window, self.ciphertext_length)
			next_bp = random.randint(low, high)
		else:
			next_cipherbet_2 = self.cipherbet_2.copy()
			next_bp = self.bp

		return Cipher(self.has_breakpoint, next_bp, next_cipherbet_1, next_cipherbet_2, self.ciphertext_length)

def initial_guess_heuristic(ciphertext: str) -> list:
	'''
	Compute and return an heuristics-based initialization for the cipher, 
	'''
	sorted_letter_probabilites = sorted(ALPHABET, key = lambda a : letter_probabilities[LETTER_TO_IDX[a]])
	count = Counter(ciphertext)
	sorted_count =  sorted(count.keys(), key = lambda a : count[a])
	sorted_count = list(set(ALPHABET).difference(sorted_count)) + sorted_count
	cipherbet = [sorted_count[sorted_letter_probabilites.index(a)] for a in ALPHABET]
	return cipherbet

def metropolis_hastings(ciphertext: str, has_breakpoint: bool, N: int) -> list:
	'''
	Use the Metropolis Hastings algorithm to approximately sample a ciphertext
		from the a posteriori distribution, given the ciphertext.

	`N` is the number of iterations that the algorithm runs for.
	`has_breakpoint` is true if and only if the ciphertext is encoded with a breakpoint.
	'''
	bp = len(ciphertext)//2 if has_breakpoint else len(ciphertext)
	initial_cipher = Cipher.create_initial_cipher(ciphertext, has_breakpoint, bp)

	samples = [initial_cipher]

	for n in range(N):
		previous_cipher = samples[-1]
		bp_sample_window = len(ciphertext)//(n//100 + 1)//10 ## reduce the breakpoint sample window as iteration count increases.

		if n % 50 == 0 and has_breakpoint:
			next_cipher = Cipher.create_initial_cipher(ciphertext, has_breakpoint, previous_cipher.bp)
		else:
			next_cipher = previous_cipher.sample_next_cipher(bp_sample_window)
		log_acceptance_factor = min(0.0, next_cipher.log_likelihood(ciphertext) - previous_cipher.log_likelihood(ciphertext))
		u = np.random.binomial(1.0, 10.0 ** log_acceptance_factor)

		if u:
			samples.append(next_cipher)
		else:
			samples.append(previous_cipher)
	
	return samples

def decoding_accuracy(plaintext: str, plaintext_estimate: str) -> float:
	'''
	Returns the accuracy given the true plaintext `plaintext` and a decoded estimate `plaintext_estimate`.
	'''
	result = 0.0
	for p, q in zip(plaintext, plaintext_estimate):
		if p == q:
			result += 1
	return result / len(plaintext)
	

def decode(ciphertext: str, has_breakpoint: bool) -> str:
	'''
	Estimate the maximum a posteriori (MAP) cipher, given `ciphertext`, and return the corresponding decoded ciphertext.
	'''
	N = 1000
	samples = metropolis_hastings(ciphertext, has_breakpoint, N)
	map_estimate = max(samples, key = lambda sample : sample.log_likelihood(ciphertext))
	plaintext = map_estimate.invert(ciphertext)
	return plaintext

def main():
	with open('./ciphertext_bp.txt', 'r') as ciphertext_file:
		ciphertext = ciphertext_file.read()
		ciphertext = ciphertext.rstrip("\r\n")
		ciphertext_file.close()

	with open('./plaintext.txt', 'r') as plaintext_file:
		plaintext = plaintext_file.read()
		plaintext = plaintext.rstrip("\r\n")
		plaintext_file.close()

	plaintext_estimate = decode(ciphertext, False)

	print(decoding_accuracy(plaintext, plaintext_estimate))

if __name__ == '__main__':
	main()