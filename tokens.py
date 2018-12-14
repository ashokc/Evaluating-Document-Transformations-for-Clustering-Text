
# import modules & set up logging
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import numpy as np

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

__all__ = ['Tokens']

class Tokens():

    es = Elasticsearch([{'host':'localhost','port':9200}])
    es_logger = logging.getLogger('elasticsearch')
    es_logger.setLevel(logging.WARNING)

    def __init__(self, dataSource):
        if (dataSource == 'twenty-news'):
            self.esIndex = 'twenty-news'
        elif (dataSource == 'acl-imdb'):
            self.esIndex = 'acl-imdb'

    def getTokens(self,tokenType, groupIndex):
        X, classNames = [], []
        docType = 'article'
        query = { "query": { "term" : {"groupIndex" : groupIndex} }, "_source" : [tokenType, 'groupName'] }
        hits = scan (self.es, query=query, index=self.esIndex, doc_type=docType, request_timeout=120)
        for hit in hits:
            X.append(hit['_source'][tokenType])
            classNames.append(hit['_source']['groupName'])
        return X, classNames[0]

