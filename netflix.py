import pandas as pd
import numpy as np
import pickle
from surprise import Reader, Dataset, SVD, evaluate, SVDpp, SlopeOne, NMF, NormalPredictor, BaselineOnly, CoClustering

#Reading Datafile
df = pd.read_csv(r"C:\Users\risha\Desktop\kaggle\combined_data_1.txt", header = None, names=["Cust_id", "Rating"], usecols=[0,1])
df["Rating"] = df["Rating"].astype(float)
movie_count = df.isnull().sum()[1]
cust_count = df["Cust_id"].nunique()
rating_count = df["Rating"].count() - movie_count


#Reading Movie Titles
df_title = pd.read_csv(r"C:\Users\risha\Desktop\kaggle\movie_titles.csv", encoding = "ISO-8859-1", header = None, names=["Movie_id", "Year", "Name"])
df_title.set_index("Movie_id", inplace=True)


#Plot to see the distribution of ratings
p = df.groupby('Rating')['Rating'].agg(['count'])
ax = p.plot(kind="barh", figsize = (15,10))
for i in range(5):
    ax.text(p.iloc[i][0]/4, i, round(p.iloc[i][0]*100/p.sum()[0]), color = "white", weight = "bold")


#Data Cleaning and manipulation
df_nan = pd.DataFrame(pd.isnull(df.Rating))
df_nan = df_nan[df_nan['Rating'] == True]
df_nan = df_nan.reset_index()
movie_np = []
movie_id = 1

for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):
    temp = np.full((1,i-j-1), movie_id)
    movie_np = np.append(movie_np, temp)
    movie_id += 1

last_record = np.full((1,len(df) - df_nan.iloc[-1, 0] - 1),movie_id)
movie_np = np.append(movie_np, last_record)

df = df[pd.notnull(df['Rating'])]

df["Movie_id"] = movie_np.astype(int)
df['Cust_id'] = df['Cust_id'].astype(int)


#Data Slicing to make the algorithm faster and avoid memory error
df_movie_summary = df.groupby('Movie_id')['Rating'].agg(["count","mean"])
df_movie_summary.index = df_movie_summary.index.map(int)
movie_benchmark = round(df_movie_summary['count'].quantile(0.7),0)
drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

df_cust_summary = df.groupby('Cust_id')['Rating'].agg(["count","mean"])
df_cust_summary.index = df_cust_summary.index.map(int)
cust_benchmark = round(df_cust_summary['count'].quantile(0.7),0)
drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

df = df[~df['Movie_id'].isin(drop_movie_list)]
df = df[~df['Cust_id'].isin(drop_cust_list)]

#Pivot data
df_p = pd.pivot_table(df, index="Cust_id", columns="Movie_id", values="Rating")

#See which algorithm gives the lowest RMSE value
reader = Reader()
data = Dataset.load_from_df(df[['Cust_id', 'Movie_id', 'Rating']][:100000], reader)
benchmark = []
for algo in [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), BaselineOnly(), CoClustering()]:
    data.split(n_folds=3)
    results = evaluate(algo, data, measures = ["RMSE"])
    
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algo).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)

print(pd.DataFrame(benchmark).set_index('Algorithm').sort_values('rmse'))

##Train and Test split
#reader = Reader()
#data = Dataset.load_from_df(df[['Cust_id', 'Movie_id', 'Rating']], reader)
#trainset, testset = train_test_split(data, test_size = 0.25)
#blo = BaselineOnly()
#blo.fit(trainset)
#predictions = blo.test(testset[:10000])
#accuracy.rmse(predictions)

#SVD gave the lowest RMSE, so we proceed with that
reader = Reader()
svd = SVD()
data = Dataset.load_from_df(df[['Cust_id', 'Movie_id', 'Rating']], reader)
trainset = data.build_full_trainset()
svd.train(trainset)

#Save model in pickle
save_model = pickle.dump(svd, open(r"C:\Users\risha\Desktop\kaggle\svdmodel.sav", "wb"))

#Load mode from pickle
svd_pickle = pickle.load(open(r"C:\Users\risha\Desktop\kaggle\svdmodel.sav", "rb"))

#Predict the movies a user will like
def user_movies(userid):
    user = df_title.copy()
    user = user.reset_index()
    user = user[~user['Movie_id'].isin(drop_movie_list)]  
    user["Estimate_Score"] = user["Movie_id"].apply(lambda x: svd_pickle.predict(userid, x).est)
    user = user.drop("Movie_id", axis=1)
    user = user.sort_values("Estimate_Score", ascending = False)
    print(user.head(10))


#Recommend similar movies (Pearson's Correlation)
def similar_movies(movie_title):
    print("For movie: " + str(movie_title))
    print("Top 10 movies recommended are: ")
    i = int(df_title.index[df_title["Name"] == movie_title][0])
    target = df_p[i]
    similar_to_target = df_p.corrwith(target)
    corr_target = pd.DataFrame(similar_to_target, columns=["PearsonR"])
    corr_target.dropna(inplace = True)
    corr_target = corr_target.sort_values("PearsonR", ascending = False)
    corr_target = corr_target.join(df_title)[["PearsonR", "Name"]]
    corr_target = corr_target.iloc[1:]
    print(corr_target[:10].to_string(index=False))


#Movies similar to Screamers
similar_movies("Screamers")

#Movies user_785314 may like
user_movies(785314)












