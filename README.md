
![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg) <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

This is an app for recommending movies based from users selecting movies they already like. [**Try the app here.**](https://share.streamlit.io/tomukmatthews/film-recommender-app/src/app/ui.py)

The recommendations are from a hybrid content / collaborative based recommender system. It uses text based metadata from the films (content based) such as cast, director, keywords and overview descriptive text. It then applies NLP methods (TF-IDF) to compute features describing each film. It also considers the films liked by users with similar preferences (collaborative based).

Dimensionality reduction via singular value decomposition is performed on the set of user-film interactions to create a representative set of users that encode the preferences of the population.

Each of these approaches are used to construct similarity matrices, with the rows and columns being the films and the numbers being how similar each film is to another on a scale of 0-1. The net similarity is given by the average of these similarity matrices and the parameter controls allow you to define how much weighting to give each method of similarity.

<br />

![Alt text](images/app_demo.png?raw=true)