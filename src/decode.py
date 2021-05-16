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

path_to_letter_probabilities = './data/letter_probabilities.csv'
path_to_transition_probabilties = './data/letter_transition_matrix.csv'

# acceptance_rate_window = 50.0
bp_sample_window = 100 # may want this to be proportional to the length of the ciphertext

with open(path_to_letter_probabilities) as letter_prob_file:
	letter_probabilities = list(csv.reader(letter_prob_file))[0]
letter_probabilities = [float(prob) for prob in letter_probabilities]

with open(path_to_transition_probabilties) as transition_file:
	letter_transition_matrix = list(csv.reader(transition_file))
letter_transition_matrix = [[float(prob) for prob in letter_transition_matrix[i]] for i in range(len(letter_transition_matrix))]


class Cipher:

	def __init__(self, has_breakpoint: bool, bp: int, cipherbet_1: str, cipherbet_2: str, ciphertext_length: int):
		self.has_breakpoint = has_breakpoint
		self.bp = bp
		self.cipherbet_1 = cipherbet_1
		self.cipherbet_2 = cipherbet_2
		self.ciphertext_length = ciphertext_length

	@classmethod
	def create_initial_cipher(cls, ciphertext: str, has_breakpoint: bool):
		bp = len(ciphertext) // 2 if has_breakpoint else len(ciphertext)

		cipherbet_1 = initial_guess_heuristic(ciphertext[:bp])
		cipherbet_2 = initial_guess_heuristic(ciphertext[bp:])

		ciphertext_length = len(ciphertext)

		return cls(has_breakpoint, bp, cipherbet_1, cipherbet_2, ciphertext_length)

	def invert(self, ciphertext: str) -> str:
		CIPHERBET_1_LETTER_TO_ID = dict(map(reversed, enumerate(self.cipherbet_1)))
		plaintext_1 = "".join(ALPHABET[CIPHERBET_1_LETTER_TO_ID[c]] for c in ciphertext[:self.bp])

		if not self.has_breakpoint:
			return plaintext_1

		CIPHERBET_2_LETTER_TO_ID = dict(map(reversed, enumerate(self.cipherbet_2)))
		plaintext_2 = "".join(ALPHABET[CIPHERBET_2_LETTER_TO_ID[c]] for c in ciphertext[self.bp:])

		return plaintext_1 + plaintext_2

	def log_likelihood(self, ciphertext: str) -> float:
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

# def invert_cipher(ciphertext: str, cipherbet: list) -> str:
# 	CIPHERBET_LETTER_TO_ID = dict(map(reversed, enumerate(cipherbet)))
# 	plaintext = "".join(ALPHABET[CIPHERBET_LETTER_TO_ID[c]] for c in ciphertext)
# 	return plaintext

# def sample_swap(cipherbet: list) -> list:
# 	i, j = random.sample(range(len(cipherbet)), 2)
# 	next_cipherbet = cipherbet.copy()
# 	next_cipherbet[i], next_cipherbet[j] = cipherbet[j], cipherbet[i]
# 	return next_cipherbet


# def log_likelihood(ciphertext: str, cipherbet: list) -> float:
# 	plaintext = invert_cipher(ciphertext, cipherbet)
# 	result = 0.0
# 	for i, p in enumerate(plaintext):
# 		if i==0:
# 			probability = letter_probabilities[LETTER_TO_IDX[p]]
# 			result += log(probability if probability != 0 else 0.000001, 10.0)
# 		else:
# 			q = plaintext[i-1]
# 			probability = letter_transition_matrix[LETTER_TO_IDX[p]][LETTER_TO_IDX[q]]
# 			result += log(probability if probability != 0 else 0.000001, 10.0)
# 	return result

def initial_guess_heuristic(ciphertext: str) -> list:
	sorted_letter_probabilites = sorted(ALPHABET, key = lambda a : letter_probabilities[LETTER_TO_IDX[a]])
	count = Counter(ciphertext)
	sorted_count =  sorted(count.keys(), key = lambda a : count[a])
	sorted_count = list(set(ALPHABET).difference(sorted_count)) + sorted_count
	cipherbet = [sorted_count[sorted_letter_probabilites.index(a)] for a in ALPHABET]
	return cipherbet

def metropolis_hastings(ciphertext: str, has_breakpoint: bool, N: int) -> list:
	# random initial cipher
	# cipherbet = ALPHABET.copy()
	# random.shuffle(cipherbet)

	# heuristic cipher depending on letter_probabilities and frequency counts
	
	initial_cipher = Cipher.create_initial_cipher(ciphertext, has_breakpoint)

	samples = [initial_cipher]

	# log_likelihoods = []
	# acceptance_rates = []
	# acceptances = 0.0
	# decoding_accuracies = []

	for n in range(N):
		# if (n % acceptance_rate_window == 0 and n != 0) or n == N-1:
		# 	acceptance_rates.append(acceptances / acceptance_rate_window)
		# 	acceptances = 0.0
		previous_cipher = samples[-1]
		bp_sample_window = len(ciphertext)//(n//100 + 1)//10
		next_cipher = previous_cipher.sample_next_cipher(bp_sample_window)
		log_acceptance_factor = min(0.0, next_cipher.log_likelihood(ciphertext) - previous_cipher.log_likelihood(ciphertext))
		u = np.random.binomial(1.0, 10.0 ** log_acceptance_factor)

		if u:
			samples.append(next_cipher)
			# acceptances += 1
		else:
			samples.append(previous_cipher)

		# if n % 100 == 0:
		# 	print(previous_cipher.bp)
		# 	print(previous_cipher.log_likelihood(ciphertext))
		# log_likelihoods.append(log_likelihood(ciphertext, samples[-1])/float(len(ciphertext)))
		# decoding_accuracies.append(decoding_accuracy(samples[-1]))

	# with open('./data/sample/d_quarter.txt', 'w') as file:
	# 	# ll_file.write(str(log_likelihoods))
	# 	file.write(str(decoding_accuracies))
	
	return samples

# def decoding_accuracy(cipherbet: list) -> float:
# 	with open('./data/sample/ciphertext.txt', 'r') as ciphertext_file:
# 		ciphertext = ciphertext_file.read()
# 		ciphertext = ciphertext.rstrip("\r\n")
# 		ciphertext_file.close()
# 	with open('./data/sample/plaintext.txt', 'r') as plaintext_file:
# 		plaintext = plaintext_file.read()
# 		plaintext = plaintext.rstrip("\r\n")
# 		plaintext_file.close()
# 	result = 0.0
# 	for c,p in zip(ciphertext, plaintext):
# 		if cipherbet[LETTER_TO_IDX[p]] == c:
# 			result += 1
# 	return result / len(plaintext)

# def decode_single_ciphertext(ciphertext: str, N: int) -> str:
	

def decode(ciphertext: str, has_breakpoint: bool) -> str:
	if has_breakpoint:
		N = 10000
	else:
		N = 5000
	samples = metropolis_hastings(ciphertext, has_breakpoint, N)
	map_estimate = max(samples, key = lambda sample : sample.log_likelihood(ciphertext))
	plaintext = map_estimate.invert(ciphertext)
	return plaintext

	# if has_breakpoint:
	# 	plaintexts = []
	# 	for i in range(len(ciphertext)):
	# 		ciphertext1, ciphertext2 = ciphertext[:i], ciphertext[i:]
	# 		plaintexts.append(decode_single_ciphertext(ciphertext1, 200) + decode_single_ciphertext(ciphertext2, 200))
	# 	return max(plaintexts, key = lambda plaintext : log_likelihood(plaintext, ALPHABET))
	# else:
	# 	return decode_single_ciphertext(ciphertext, N)

def main():
	# path_to_letter_probabilities = '../data/letter_probabilities.csv'
	# path_to_transition_probabilties = '../data/letter_transition_matrix.csv'

	with open('./data/sample/ciphertext.txt', 'r') as ciphertext_file:
		ciphertext = ciphertext_file.read()
		ciphertext = ciphertext.rstrip("\r\n")
		ciphertext_file.close()

	plaintext_part_1= decode(ciphertext, False)

	with open('./data/sample/plaintext_part_1.txt', 'w') as output_file:
		output_file.write(plaintext_part_1)

if __name__ == '__main__':
	main()