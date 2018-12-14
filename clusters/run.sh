#!/bin/bash

#wordCorpus="twenty-news"
#wordCorpus="acl-imdb"
wordCorpus=$1
min_df=2
for tokenType in stopped; do
	for wordVecSource in none svd glove fasttext word2vec custom-vectors-fasttext custom-vectors-word2vec; do
			runStart=`date +%s`
			if [ "$wordCorpus" == "twenty-news" ] ; then
				classes="3,10,15"
			fi ;
			if [ "$wordCorpus" == "acl-imdb" ] ; then
				classes="0,1"
			fi ;
			echo "pipenv run python cluster.py $wordCorpus $min_df $tokenType $wordVecSource $classes"
			pipenv run python cluster.py $wordCorpus $min_df $tokenType $wordVecSource $classes
			runEnd=`date +%s`
			runtime=$((runEnd-runStart))
			outfile=$wordCorpus"-"$tokenType"-"$wordVecSource
			echo "Time taken for results/$outfile: $runtime"
			mv results.csv results/$outfile.csv
#			mv logs/clusters.log results/$outfile.log
	done
done
