from MovieLens import MovieLens
from RBMAlgorithm import RBMAlgorithm
from ContentKNNAlgorithm import ContentKNNAlgorithm
from HybridAlgorithm import HybridAlgorithm
from surprise import NormalPredictor
from surprise import SVD, SVDpp
from Evaluator import Evaluator
import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

# Just make random recommendations
Random = NormalPredictor()
# Simple RBM
SimpleRBM = RBMAlgorithm(epochs=40)
# Content
ContentKNN = ContentKNNAlgorithm()
# SVD
SVD = SVD(n_epochs=20, lr_all=0.005, n_factors=50)
# SVD++
SVDPlusPlus = SVDpp()

#Combine them
Hybrid = HybridAlgorithm([SimpleRBM, ContentKNN, SVD, SVDPlusPlus], [0.1, 0.25, 0.3, 0.35])

evaluator.AddAlgorithm(Random, "Random")
evaluator.AddAlgorithm(ContentKNN, "ContentKNN")
evaluator.AddAlgorithm(SVD, "SVD")
evaluator.AddAlgorithm(SVDPlusPlus, "SVD++")
evaluator.AddAlgorithm(SimpleRBM, "RBM")
evaluator.AddAlgorithm(Hybrid, "Hybrid")

evaluator.Evaluate(False)

evaluator.SampleTopNRecs(ml, 85)
