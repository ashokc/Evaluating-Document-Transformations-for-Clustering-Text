import json

import sys
sys.path.append('..')

from wordvectors import WordVectors
from vectorizers import VectorizerWrapper, Transform2WordVectors
from tokens import Tokens
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix

def svdReduce (X, order=300):
    svd = TruncatedSVD(n_components=order, n_iter=7, random_state=42)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    denseZ = lsa.fit_transform(X)
    return denseZ

def getX (wordCorpus, tokenType, groupIndices):
    XAll = []
    indexList = {}
    start = 0 
    for groupIndex in groupIndices:
        X, className = Tokens(wordCorpus).getTokens(tokenType, groupIndex)
        end = start + len(X)
        indexList[className] = {'start' : start, 'end' : end}
        XAll = XAll + X
        start = end
    XAll = np.array([np.array(xi) for xi in XAll])          #   rows: Docs. columns: words
    return XAll, indexList

def getSimDist (vec1, vec2):
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    if (mag1 > 0 and mag2 > 0):
        dist = np.linalg.norm(vec1 - vec2)
        similarity = np.dot(vec1,vec2) / (mag1 * mag2)
        return dist, similarity
    else:
        return -99999.0, -99999.0

def computeMetrics (allVectors, indexList, tokenType, orderReduction):
    clusterCentroids = []
    intraClusterMetics = {}
    clusterNames = []
    for trueCluster, startEnd in indexList.items():
        clusterNames.append(trueCluster)
        thisVectors = allVectors[startEnd['start']:startEnd['end']]
        thisCentroid = np.average(thisVectors, axis=0)
        clusterCentroids.append(thisCentroid)

        intraClusterMetics[trueCluster] = {}
        similarities = []
        distances = []
        for i in range(0, len(thisVectors)):
            dist, similarity = getSimDist (thisVectors[i], thisCentroid)
            if (dist > 0.0):
                similarities.append(similarity)
                distances.append(dist)

        intraClusterMetics[trueCluster]['similarities'] = similarities
        intraClusterMetics[trueCluster]['distances'] = distances
        intraClusterMetics[trueCluster]['similarities-stats'] = {'min' : np.amin(similarities), 'mean' : np.mean(similarities), 'median' : np.median(similarities), 'max' : np.amax(similarities), 'std' : np.std(similarities)}
        intraClusterMetics[trueCluster]['distances-stats'] = {'min' : np.amin(distances), 'mean' : np.mean(distances), 'median' : np.median(distances), 'max' : np.amax(distances), 'std' : np.std(distances)}

    centroidMetrics = {}
    centroidMetrics['distances'] = {}
    centroidMetrics['distances-stats'] = {}
    centroidMetrics['similarities'] = {}
    centroidMetrics['similarities-stats'] = {}
    for clusterName in clusterNames:
        centroidMetrics['distances'][clusterName] = {}
        centroidMetrics['similarities'][clusterName] = {}

    for i in range(0, len(clusterCentroids)):
        for j in range(0, len(clusterCentroids)):
            if (i != j):
                dist, similarity = getSimDist (clusterCentroids[i], clusterCentroids[j])
                centroidMetrics['distances'][clusterNames[i]][clusterNames[j]] = dist
                centroidMetrics['similarities'][clusterNames[i]][clusterNames[j]] = similarity

        dists = np.array(list(centroidMetrics['distances'][clusterNames[i]].values()))
        centroidMetrics['distances-stats'][clusterNames[i]] = {'min' : np.amin(dists), 'mean' : np.mean(dists), 'median' : np.median(dists), 'max' : np.amax(dists), 'std' : np.std(dists)}
        sims = np.array(list(centroidMetrics['distances'][clusterNames[i]].values()))
        centroidMetrics['similarities-stats'][clusterNames[i]] = {'min' : np.amin(sims), 'mean' : np.mean(sims), 'median' : np.median(sims), 'max' : np.amax(sims), 'std' : np.std(sims)}

    return intraClusterMetics, centroidMetrics

def main (wordCorpus):
    min_df = 2
    if (wordCorpus == 'twenty-news'):
        groupIndices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        orderReductions = ['none',  'svd', 'glove', 'fasttext', 'word2vec', 'custom-vectors-fasttext', 'custom-vectors-word2vec']
    elif (wordCorpus == 'acl-imdb'):
        groupIndices = [0,1]
        orderReductions = ['svd', 'glove', 'fasttext', 'word2vec', 'custom-vectors-fasttext', 'custom-vectors-word2vec']

    nClusters = len(groupIndices)

    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=min_df)

    metrics = {}
    for tokenType in ['stopped']:
        X, indexList = getX(wordCorpus, tokenType, groupIndices)

        sparseX = vectorizer.fit_transform(X)
        corpusVocab = vectorizer.vocabulary_

        metrics[tokenType] = {}

        for orderReduction in orderReductions:
            if ( ( (orderReduction == 'word2vec') or (orderReduction == 'fasttext') or (orderReduction == 'glove') ) and (tokenType != 'stopped') ):
                continue
            else:
                print (tokenType, orderReduction)
                metrics[tokenType][orderReduction] = {}

                if ( (orderReduction != 'svd') and (orderReduction != 'none') ):
                    wvObject = WordVectors(wordCorpus=wordCorpus, wordVecSource=orderReduction,corpusVocab=corpusVocab,tokenType=tokenType)

                if (orderReduction == 'none'):
                    denseZ = sparseX
                    denseZ = denseZ.toarray()
                elif (orderReduction == 'svd'):
                    denseZ = svdReduce (sparseX, order=300)
                else:
                    argsForTransform = { 'sparseX' : sparseX, 'vocab' : corpusVocab }
                    denseZ = Transform2WordVectors(wvObject).transform(argsForTransform)

                normalizer = Normalizer(copy=False)
                denseZ = normalizer.fit_transform(denseZ)
                intraClusterMetics, centroidMetrics = computeMetrics (denseZ, indexList, tokenType, orderReduction)
                metrics[tokenType][orderReduction]['intraClusterMetics'] = intraClusterMetics
                metrics[tokenType][orderReduction]['centroidMetrics'] = centroidMetrics

    f = open ('./results/' + wordCorpus + '.json','w')
    out = json.dumps(metrics, ensure_ascii=True)
    f.write(out)
    f.close()

if __name__ == '__main__':
    args = sys.argv
    wordCorpus = str(args[1])
    main(wordCorpus)

