import numpy as np
import sys

import matplotlib.pyplot as plt
from PIL import Image
import json

#get_ipython().magic('matplotlib inline')

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def plotWhiskers (metrics):
    group = 'alt.atheism'
    dboxes_original = [ metrics['stopped']['none']['intraClusterMetics'][group]['distances'], list(metrics['stopped']['none']['centroidMetrics']['distances'][group].values()) ]
    dboxes_transformed = [ metrics['stopped']['custom-vectors-word2vec']['intraClusterMetics'][group]['distances'], list(metrics['stopped']['custom-vectors-word2vec']['centroidMetrics']['distances'][group].values()) ]

    ticks = ['Intracluster', 'Intercluster']

    b1 = plt.boxplot(dboxes_original, widths=[0.125, 0.125], positions=[1, 2], showmeans=True,sym='',whis=[5,95])
    b2 = plt.boxplot(dboxes_transformed, widths=[0.125, 0.125], positions=[1.25, 2.25], showmeans=True,sym='',whis=[5,95])
    set_box_color(b1, 'b')
    set_box_color(b2, 'r')
    plt.plot([], c='b', label='')
    plt.plot([], c='r', label='')
#    plt.legend(fontsize=8)

    plt.xticks([1.125, 2.125], ticks)
    plt.xlim(0.85, 2.4)
    plt.ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig('results/' + group + '_whiskers_transformed.png', format='png', dpi=720)
    plt.close()

def plotBars(wordCorpus, sortedRatios, metric, tokenType):
    fig = plt.figure(figsize=(6,6),dpi=720)
    subplot = fig.add_subplot(1, 1, 1)
    vals = [item[1] for item in sortedRatios]
    labels = [item[0] for item in sortedRatios]
    xLocs = np.linspace(0.1, 0.9, len(vals))
    width = 0.8 / len(vals) *0.5
    if (wordCorpus == 'acl-imdb'):
        colors = ['g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
    elif (wordCorpus == 'twenty-news'):
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']

    subplot.bar(xLocs,vals,width=width,color=colors,tick_label=labels)
    fig.savefig('./results/' + wordCorpus + '-' + metric + '-' + tokenType + '.png', format='png', dpi=720)

def main(wordCorpus):
    f = open ('./results/' + wordCorpus + '.json','r')
    metrics = json.loads(f.read())
    f.close()

    groups = metrics['stopped']['svd']['centroidMetrics']['distances'].keys()
    nGroups = len(groups)

    if (wordCorpus == 'twenty-news'):
        labels = ['none',  'svd', 'glove', 'fast', 'w2v', 'cv-fast', 'cv-w2v']
        orderReductions = ['none',  'svd', 'glove', 'fasttext', 'word2vec', 'custom-vectors-fasttext', 'custom-vectors-word2vec']
        plotWhiskers (metrics)
    elif (wordCorpus == 'acl-imdb'):
        labels = ['svd', 'glove', 'fast', 'w2v', 'cv-fast', 'cv-w2v']
        orderReductions = ['svd', 'glove', 'fasttext', 'word2vec', 'custom-vectors-fasttext', 'custom-vectors-word2vec']

    results = []
    for metric in ['distances', 'similarities']:
        for tokenType in ['stopped']:
            ratioResults = {}
            for i, orderReduction in enumerate(orderReductions):
                if ( ( (orderReduction == 'word2vec') or (orderReduction == 'fasttext') or (orderReduction == 'glove') ) and (tokenType != 'stopped') ):
                    continue
                marker = tokenType + '-' + orderReduction + '-' + metric

                avgMedian = 0.0
                for group in groups:
                    avgMedian = avgMedian + metrics[tokenType][orderReduction]['intraClusterMetics'][group][metric+'-stats']['median']
                avgMedian = avgMedian / nGroups

                avgIntercluster = 0.0
                count = 0
                for group in groups:
                    count = count + len(metrics[tokenType][orderReduction]['centroidMetrics'][metric][group])
                    centroidMetrics = metrics[tokenType][orderReduction]['centroidMetrics'][metric][group]
                    avgIntercluster = avgIntercluster + sum(centroidMetrics.values())
                print ("inter cluster count Vs ng*(ng-1)", count, nGroups * (nGroups - 1))
                avgIntercluster = avgIntercluster / (nGroups * (nGroups - 1))
                ratio = avgIntercluster / avgMedian
                ratioResults[labels[i]] = ratio
                results.append( [tokenType, orderReduction, metric, avgMedian, avgIntercluster, ratio] )

            reverse = False
            if (metric == 'similarities'):
                reverse = True
            sortedRatios = sorted(ratioResults.items(), key=lambda kv: kv[1],reverse=reverse)
            print ('\t As-Is',ratioResults)
            print ('\t Sorted',sortedRatios)
            plotBars(wordCorpus, sortedRatios, metric, tokenType)

    header = 'tokenType, orderReduction, metric, Avg. Intracluster Median - aim, Avg Intercluster - ai, Ratio - ai/aim'
    np.savetxt('results/' + wordCorpus + '-results.csv', results, header=header,delimiter=',',fmt='%s') 

if __name__ == '__main__':
    args = sys.argv
    wordCorpus = str(args[1])
    main(wordCorpus)

