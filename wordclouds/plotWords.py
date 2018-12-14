from wordcloud import WordCloud
import sys
sys.path.append('..')
from tokens import Tokens
#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

def main (wordCorpus):
    min_df = 2
    tokenType = 'stopped'
    if (wordCorpus == 'twenty-news'):
        groupIndices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    elif (wordCorpus == 'acl-imdb'):
        groupIndices = [0,1]
    nClusters = len(groupIndices)
    for groupIndex in groupIndices:
        tokensLists, className = Tokens(wordCorpus).getTokens(tokenType, groupIndex)
        flat_list = [tokens for tokensList in tokensLists for tokens in tokensList]
        text = ' '.join(flat_list)
        wordcloud = WordCloud(max_font_size=40, width=600, height=400, background_color='white', max_words=200, relative_scaling=1.0).generate_from_text(text)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        wordcloud.to_file('./results/' + className + '.jpg')

if __name__ == '__main__':
    args = sys.argv
    wordCorpus = str(args[1])
    main(wordCorpus)

