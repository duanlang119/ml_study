# 第二步： 读取全部训练集：
from re import sub

from jieba import cut

allWords=[]
def getWordsFromFile(textFile):
    words=[]
    with open(textFile,encoding='utf-8') as fp:
        for line in fp:
            line=line.strip()
            # 过滤干扰字符或无效字符
            line=sub(r'[,【】0-9、-。，！~\*]','',line)
            line=cut(line)
            line=filter(lambda word:len(word)>1,line)
            words.extend(line)

    return words

def getTopNWords(topN):
    pass