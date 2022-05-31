from typing import List, Callable, Iterable
from dataclasses import dataclass, fields
from enum import Enum, auto
from nltk import PorterStemmer, word_tokenize, WordNetLemmatizer, edit_distance
from nltk.corpus import stopwords, words
from pandas import DataFrame
import pandas as pd
import nltk

print(nltk.data.path)

nltk.data.path.append('./nltk_data')

nltk.download('stopwords')
nltk.download('words')

stopwords = set(stopwords.words('english'))
words = set(words.words('en'))


class Sentiment(Enum):
    NEGATIVE = 'negative'
    NEUTRAL = 'neutral'
    POSITIVE = 'positive'


@dataclass
class Tweet:
    tweet: str
    sentiment: Sentiment


@dataclass
class StemEntry:
    stem: str
    negative: int
    positive: int
    neutral: int
    weight: int

    @classmethod
    def default(cls, stem: str):
        return cls(
            stem=stem,
            negative=0,
            positive=0,
            neutral=0,
            weight=0
        )

class Analysis:
    negative: float = 0
    positive: float = 0
    neutral: float = 0

    def __init__(self, tokens: Iterable[StemEntry]):
        tokens = list(tokens)
        self.total_weight = sum(token.weight for token in tokens)
        for token in tokens:
            self.positive += self.__prob(token.positive, token.weight)
            self.neutral += self.__prob(token.neutral, token.weight)
            self.negative += self.__prob(token.negative, token.weight)

    def __str__(self):
        return (f"Sentiment bayesian analysis result: {self.sentiment}\n"
              f"- negative matches: {self.negative}\n"
              f"- positive matches: {self.positive}\n"
              f"- neutral matches: {self.neutral}")

    @property
    def sentiment(self) -> Sentiment:
        max_metric = max((self.neutral, self.negative, self.positive))
        if max_metric == self.neutral:
            return Sentiment.NEUTRAL
        elif max_metric == self.positive:
            return Sentiment.POSITIVE
        else:
            return Sentiment.NEGATIVE

    def __prob(self, amount: int, weight: int):
        return float(int(amount) * int(weight))


class TwitterClassifier:

    def __init__(self, **opts):
        self.tokenizer = opts['tokenizer'] if 'tokenizer' in opts else nltk.RegexpTokenizer(r"\w+")
        self.correction = opts['correction'] if 'correction' in opts else False

        self.stemmer = PorterStemmer()
        self.words = DataFrame(columns=[field.name for field in fields(StemEntry)])

    def train(self, data: DataFrame):
        for index, row in data.iterrows():
            print(row)
            row = Tweet(tweet=row['tweet'], sentiment=row['sentiment'])
            tokenstream = self.tokenize(row.tweet)
            sentiment = row.sentiment
            for token in tokenstream:
                if len(self.words.loc[self.words['stem'] == token]) == 0:
                    self.words = pd.concat([self.words, DataFrame([StemEntry.default(token)])],
                        axis=0,
                        join='outer'
                    )
                self.words.loc[self.words['stem'] == token, 'weight'] += 1
                if sentiment == Sentiment.NEGATIVE:
                    self.words.loc[self.words['stem'] == token, 'negative'] += 1
                elif sentiment == Sentiment.POSITIVE:
                    self.words.loc[self.words['stem'] == token, 'positive'] += 1
                else:
                    self.words.loc[self.words['stem'] == token, 'neutral'] += 1

    def test(self, data: DataFrame):
        return sum(1
                   for index, row in data.iterrows()
                   if self.analyse(row['tweet']).sentiment == row['sentiment']) / data.shape[0]

    def tokenize(self, tweet: str) -> List[str]:
        tokens = self.tokenizer.tokenize(tweet)
        tokens = [token.lower() for token in tokens]
        if self.correction:
            tokens = [token if token in words else self.correct(token) for token in tokens]
        tokens = [self.stemmer.stem(token)
                  for token in tokens if token not in stopwords]
        print(tokens)
        return tokens

    def analyse(self, tweet: str) -> Analysis:
        tokenstream = self.tokenize(tweet)
        return Analysis(StemEntry(**row) for token in tokenstream if len(row := self.words.loc[self.words['stem'] == token]) > 0)

    def correct(self, token: str) -> str:
        d = {word: edit_distance(token, word) for word in words}
        return min(d, key=d.get)
