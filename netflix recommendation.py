import pandas as pd
import numpy as np
movies = pd.read_csv(r"netflix_titles.csv")
movies.head(1)
#we only take those column which ae necessary for our prediction
movies=movies[['title','listed_in','description']]
movies
movies.isnull().sum()
#checking if any duplicate data over here
movies.duplicated().sum()
#converting listed_in  and description are string converting it in list
movies['description']=movies['description'].apply(lambda x:x.split())
movies['listed_in']=movies['listed_in'].apply(lambda x:x.split())
#now we can see that every thing is in list.
movies.head()
#now we will concatinate these two list and then make it to string.
movies['tags'] = movies['listed_in'] + movies['description']
movies.head()
new_movies = movies[['title','tags']]
new_movies.head()
#now we are again converting the list into string for further process
new_movies['tags']=new_movies['tags'].apply(lambda x: " ".join(x))
new_movies.head()
new_movies['tags'][0]
new_movies['tags'] = new_movies['tags'].apply(lambda x:x.lower())
#now we count the vector which means converting a given paragraph into a vectorized form.
from sklearn.feature_extraction.text import CountVectorizer
#first we will take maxfeature and max value to calculate the max number of words come up + stopwords too.
cv = CountVectorizer(max_features=5000, stop_words='english')
#now we use cv obj
vector = cv.fit_transform(new_movies['tags']).toarray()
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(vector).shape
similarity = cosine_similarity(vector)
new_movies[new_movies['title'] == 'Ganglands'].index[0]
#for creating a better over we just use enumerate which means we are now holding the index of  movie name
list(enumerate(similarity[0]))
#now we will sort the similarity in a sorted form and plus in reverse too so we can get those value which are more similar.
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])
#now in order to get only 5 recommendation we use index value
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]
def recommend(movie):
    index = new_movies[new_movies['title'] == movies].index[0]
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    
    
    for i in movies_list:
        print(new_movies.iloc[i[0]].title)
        #print(i[0]) this is for  printing the index value of recommended movie.
        #as we can see that it showing only indexes so we will change the index value to movie name
recommend('Ganglands')
import joblib
joblib.dump(recommend,"Recommendation_Model")
load_model=joblib.load("Recommendation_Model")

