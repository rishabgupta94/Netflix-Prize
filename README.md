# Netflix-Prize

## Dataset

- **combined_data.txt** - The first line in the dataset contains a movie id followed by a colon. Each subsequent line corresponds to a rating from a customer and it's date in the following format: CustomerID, Rating, Date. The dataset contains 4499 movies.

- **movie_titles.csv** - Movie information is in the following format: MovieId, YearOfRelease, Title


## Goal
There are two goals to this project:

1. To recommend new movies to the users with the given customer ids
2. To find similar movies, given a movie name

## Approach
Python's scikit-surprise library was used to predict the rating of the movies using the SVD algorithm (RMSE = 0.98). Pearson's Correlation was used to find the correlation between movies and further recommend similar movies.

The machine learning model was serialized and saved in a .sav file using pickle. The pickle file can be loaded to make future predictions without the need to train the model again.
