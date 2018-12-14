# Evaluating Document Transformations for Clustering Text

This is the source code to go along with the blog article  

[Clustering Text with Transformed Document Vectors](http://xplordat.com/2018/11/26/clustering-text-with-transformed-document-vectors/)

## Dependencies

	numpy
	elasticsearch
	nltk
	gensim
	scikit-learn
	wordcloud
	image
	matplotlib
	pyyaml

## Usage

### 1. Word Clouds

	cd wordclouds

	python ./plotWords.py twenty-news

to generate imges like:

![rec.sport.hockey articles](./images/rec.sport.hockey.jpg "rec.sport.hockey")

(or)

	python ./plotWords.py acl-imdb 

to generate imges like:

![Negative Reviews](./images/neg.jpg "Negative Reviews")

### 2. Intra & Inter cluster distance transformations

	cd analysis
	python ./analyze.py twenty-news
	python ./processAnalysis.py twenty-news

to generate the box-whisker plot:
![Transformation of distances](./images/alt.atheism_whiskers_transformed.png "Transformation of distances")

and for the intercluster/intracluster ratio:
![20-news B/A Ratio](./images/twenty-news-distances-stopped.png "20-news B/A Ratio")

and

	python ./analyze.py acl-imdb
	python ./processAnalysis.py acl-imdb

for the intercluster/intracluster ratio:
![Movie-Reviews B/A Ratio](./images/acl-imdb-distances-stopped.png "Moview-Reviews B/A Ratio")

### 3. Clustering

	cd clusters
	mkdir logs
	./run.sh twenty-news
	./run.sh acl-imdb

Processing the results yields images like:
![20-news Results](./images/20-news-results.jpg "20-news clustering results")

and
![Movie Reviews Results](./images/movie-reviews-results.jpg "Movie reviews clustering results")



