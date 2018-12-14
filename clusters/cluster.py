import logging
import initLogs

import sys
sys.path.append('..')

from wordvectors import WordVectors
from vectorizers import VectorizerWrapper, Transform2WordVectors
from tokens import Tokens
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix

import time

def processArgs():
    args = sys.argv
    suppliedArgs = len(args) - 1
    requiredArgs = 5
    if (suppliedArgs != requiredArgs):
        logger.critical ('Need 4 args: wordCorpus, min_df, tokenType, orderReduction, listOfClasses ... Exiting')
        sys.exit(0)
    else:
        wordCorpus = str(args[1])       # twenty-news
        min_df = int(args[2])           # 2
        tokenType = str(args[3])        # stemmed, stopped
        orderReduction = str(args[4])    # none, svd, custom-fasttext, custom-word2vec, fasttext, word2vec, glove
        listOfClasses = str(args[5])    # 3,10,15
    
        if (orderReduction == 'none'):
            orderReduction = None

        if ( ( (orderReduction == 'word2vec') or (orderReduction == 'fasttext') or (orderReduction == 'glove') ) and (tokenType != 'stopped') ):
            logger.error('For generic embedding use stopped words only... exiting')
            sys.exit(0)

    return wordCorpus, min_df, tokenType, orderReduction, listOfClasses

def svdReduce (X, order=300):
    svd = TruncatedSVD(n_components=order, n_iter=7, random_state=42)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    denseZ = lsa.fit_transform(X)
    logger.info('SVD Explained Variance Ratio:{}'.format(svd.explained_variance_ratio_))
    logger.info('SVD Sum Explained Variance Ratio:{}'.format(svd.explained_variance_ratio_.sum()))
#    logger.info('SVD Singular Values:{}'.format(svd.singular_values_))
    return denseZ

def getMinClusterSeparation (guessNclusters, cluster_centers):
    minClusterSeparation = 1.0e32
    dists = []
    for i in range (0, guessNclusters):
        for j in range (0, guessNclusters):
            if (i != j):
                diff = cluster_centers[i] - cluster_centers[j]
                dist = np.linalg.norm(diff)
                dists.append(dist)
                minClusterSeparation = min(minClusterSeparation, dist)
    logger.info('guessNclusters: {}, Distances: {}'.format(guessNclusters, dists))
    return minClusterSeparation

def getX (wordCorpus, tokenType, listOfClasses):
    XAll = []
    indexList = {}
    groupIndices = listOfClasses.split(',')
    start = 0 
    for groupIndex in groupIndices:
        X, className = Tokens(wordCorpus).getTokens(tokenType, groupIndex)
        end = start + len(X)
        indexList[className] = {'start' : start, 'end' : end}
        logger.info('True Group Index {}, classname: {}'.format(groupIndex, className))
        logger.info('Count {}, start - End Indices  {} , {}'.format(len(X),start, end))
        XAll = XAll + X
        start = end
    XAll = np.array([np.array(xi) for xi in XAll])          #   rows: Docs. columns: words
    logger.info('indexList{}'.format(indexList))
    return XAll, indexList

def main():
    start0 = time.time()

    wordCorpus, min_df, tokenType, orderReduction, listOfClasses = processArgs()
    classList = list(map(int, listOfClasses.split(',')))
    logger.info('Running: WordCorpus: {}, TokenType: {}, min_df: {}, orderReduction: {}, listOfClasses: {}'.format(wordCorpus, tokenType, min_df, orderReduction, classList))

#    vectorizers = [ ('counts', CountVectorizer(analyzer=lambda x: x, min_df=min_df)), ('tf-idf', TfidfVectorizer(analyzer=lambda x: x, min_df=min_df)) ]
    vectorizers = [ ('tf-idf', TfidfVectorizer(analyzer=lambda x: x, min_df=min_df)) ]

    X, indexList = getX(wordCorpus, tokenType, listOfClasses)
    out0 = [tokenType]
    for trueCluster, startEnd in indexList.items():
        out0.append(trueCluster + ':' + str(startEnd['end'] - startEnd['start']))

    vocabularyGenerator = CountVectorizer(analyzer=lambda x: x, min_df=min_df).fit(X) # This is only to generate a vocabulary with min_df
    corpusVocab = vocabularyGenerator.vocabulary_
    logger.info('Total Corpus Size: len(corpusVocab) with frequency > min_df : {}, X.shape: {}, # clusters: {}'.format(len(corpusVocab), X.shape, len(classList)))
    if ( (orderReduction) and (orderReduction != 'svd') ):
        wvObject = WordVectors(wordCorpus=wordCorpus, wordVecSource=orderReduction,corpusVocab=corpusVocab,tokenType=tokenType)

    results = []
    for name, vectorizer in vectorizers:
        logger.info('\n\nVectorizer: {}'.format(name))

        sparseX = vectorizer.fit_transform(X)
        if (not orderReduction):
            denseZ = sparseX
        elif (orderReduction == 'svd'):
            denseZ = svdReduce (sparseX, order=300)
        else:
            argsForTransform = { 'sparseX' : sparseX, 'vocab' : corpusVocab }
            denseZ = Transform2WordVectors(wvObject).transform(argsForTransform)
        nClusters = len(classList)

        normalizer = Normalizer(copy=False)
        denseZ = normalizer.fit_transform(denseZ)

        nRuns = 1
        for run in range(nRuns):
            result = []
            result = result + out0
            result = result + [name, run, orderReduction]
            model = KMeans(n_clusters=nClusters, max_iter=5000, tol=1.0e-8)
            labels = model.fit_predict (denseZ)
            logger.info('\nRun:{}'.format(run))
            for predictedCluster in range(nClusters):
                result.append(str(predictedCluster) + ':' + str(len(set(np.where(labels == predictedCluster)[0]))))

            for trueCluster, startEnd in indexList.items():
                predictedLabels = labels[startEnd['start']:startEnd['end']]
                for predictedCluster in range(nClusters):
                    count = len(set(np.where(predictedLabels == predictedCluster)[0]))
                    result.append(str(predictedCluster) + ':' + str(count))

            minClusterSeparation = getMinClusterSeparation (nClusters, model.cluster_centers_)
            ratio = model.inertia_ / minClusterSeparation
            result = result + [model.inertia_, minClusterSeparation, ratio]
            results.append(result)

    with open('./results.csv', 'wb') as fh1:
        np.savetxt(fh1, results, delimiter=", ", fmt='%s')

if __name__ == '__main__':
    initLogs.setup()
    logger = logging.getLogger(__name__)
    np.set_printoptions(linewidth=100)
    main()

