'''
File: kmeans.py
Author: SWETHA REVANUR
----------------------
Implementation of k-means clustering in Python.
'''

import random
import collections
import math
import sys
from util import *

def kmeans(examples, K, maxIters):
	'''
	examples: list of examples, each example is a string-to-double dict representing a sparse vector.
	K: number of desired clusters. Assume that 0 < K <= |examples|.
	maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
	Return: (length K list of cluster centroids,
			list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
			final reconstruction loss)
	'''
	zlist = [None]*len(examples)
	clusters = {}
	total_loss = 0
	prev_loss = float("inf")

	# initialize clusters
	for j in range(K):
		clusters[j] = examples[j]
	
	# cache dot products of examples
	cacheExamples = {}
	for i in range(len(examples)):
		cacheExamples[i] = dotProduct(examples[i], examples[i])

	# handles control flow of algorithm
	for t in range(maxIters):
		if prev_loss == total_loss:
			break

		prev_loss = total_loss
		total_loss = 0

		# cache dot products of clusters
		cacheClusters = {}
		for j in range(K):
			cacheClusters[j] = dotProduct(clusters[j], clusters[j])

		# assign cluster labels to examples
		for i in range(len(examples)):
			running_min = float("inf")
			exampleDot = cacheExamples[i]
			example = examples[i]
			for j in range(K):
				cluster =  clusters[j]
				clusterDot = cacheClusters[j]
				dist = abs(exampleDot - 2*dotProduct(example, cluster) + clusterDot) # preprocessed
				if dist < running_min:
					running_min = dist
					zlist[i] = j
			total_loss += running_min

		# update cluster values based on means
		for j in range(K):
			running_sum = {}
			count = 0.0
			for i in range(len(examples)):
				if zlist[i] == j:
					increment(running_sum, 1, examples[i])
					count += 1.0
			if count == 0.0:
				continue
			clusters[j] = {k : v/count for k,v in running_sum.iteritems()}

	return clusters, zlist, total_loss