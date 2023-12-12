# Ensemble Recommendation System

Every movie recommendation system that is popular and growing uses a single model for it’s recommendation process. It might be pretty accurate but people still express their dissatisfaction as it doesn’t really provide them with new movies that they would enjoy but merely something that either belongs to the same genre only or something unsatisfactory with a low rating.

To overcome this problem, we have incorporated `four different models` to create an `ensemble` for this recommendation system that would take more attributes into account before predicting the recommendations that someone would actually enjoy. We present a comprehensive exploration of a hybrid recommendation system, incorporating both `collaborative filtering` and `content-based filtering`.

### Hybrid Motivation

Content-based recommendations don't care about how much behaviour data you have, but they have their limits. We would want to take individual behaviour data into account as well whenever its possible. Our hybrid approach gives us the best of both worlds, with an accuracy measure that's in between the two.

### Hybrid Algorithm

The hybrid recommender system combines multiple individual recommender algorithms using weighted averaging to generate a final prediction, capitalizing on the strengths of different algorithms for improved recommendation performance.

This includes the use of `ContentKNN, RBM, SVD and SVD++`. Each algorithm is given an associated `weight` based on how well the individual algorithm performs on the dataset, for which the weighted average is calculated to produce the final rating estimates.

### Input and Output

The algorithm is trained on every user's behaviour history, the list of movie and the respective rating each user has given it. The user ID is given as the input on which the K neighbours having similar behaviour is calculated using `cosine similarity` and the estimated rating the user would give if they watched the movie. A list of all movies ranked on their estimated rating is returned and unseen movies are filtered out. The top-N movies are selected are returned as the output for the user.

### File structure

The jupyter notebook (`RecSys.ipynb`) is the combination of all the python files in one place while each python file represents the individual components of the project.
