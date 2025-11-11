from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd
from nltk.stem import SnowballStemmer

PORTUGUESE_STOPWORDS: Sequence[str] = (
    "a,à,agora,ainda,além,algo,algumas,alguns,ali,antes,ao,aos,apenas,apoio,após,assim,até,bem,boa,boas,bom,bons,breve,cada,coisa,com,como,contra,contudo,cuja,cujas,cujo,cujos,da,daquele,daqueles,das,de,demais,dentro,depois,desde,dessa,desse,desta,deste,disso,disto,do,dos,e,ela,elas,ele,eles,em,entre,era,eram,essa,essas,esse,esses,esta,está,estamos,estão,estas,estas,este,estes,eu,faz,fazer,fez,foi,for,fora,foram,forma,há,isso,isto,ja,já,lá,lhe,lhes,logo,mais,mas,me,mesma,mesmas,mesmo,mesmos,meu,meus,minha,minhas,na,naquele,naqueles,nas,nem,nesta,neste,ni,nos,nós,nossa,nossas,nostro,nosso,nossos,nunca,o,os,ou,outra,outras,outro,outros,para,pela,pelas,pelo,per,perante,pode,podem,por,porque,porquê,porém,posso,quais,qual,quando,quanto,que,quem,ser,se,são,seja,sem,sendo,seu,seus,sob,sobre,sua,suas,tal,também,tampouco,te,tem,tendo,ter,teu,teus,toda,todas,todo,todos,trás,tua,tuas,tudo,um,uma,umas,uns,vai,vamos,vão,vem,vindo,vos,vós".split(",")
)


@dataclass
class PreprocessConfig:
    text_column: str
    cleaned_column: str = "review_cleaned"
    min_tokens: int = 1
    apply_stemming: bool = True


class TextPreprocessor:
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self._stemmer = SnowballStemmer("portuguese") if config.apply_stemming else None
        self._stop_words = {word.strip() for word in PORTUGUESE_STOPWORDS if word.strip()}
        self._token_pattern = re.compile(r"[a-záéíóúâêôãõçàèìòùäëïöüñ]+", re.UNICODE)

    def clean_text(self, text: str | float) -> str:
        if not isinstance(text, str):
            return ""
        lowered = text.lower()
        tokens = self._token_pattern.findall(lowered)
        filtered: Iterable[str] = (
            token for token in tokens if token and token not in self._stop_words
        )
        if self._stemmer:
            filtered = (self._stemmer.stem(token) for token in filtered)
        return " ".join(filtered)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.text_column not in df.columns:
            raise KeyError(f"Column '{self.config.text_column}' not in dataframe")

        df = df.copy()
        df[self.config.cleaned_column] = df[self.config.text_column].apply(self.clean_text)
        df["review_length"] = df[self.config.cleaned_column].apply(lambda x: len(x.split()))
        if self.config.min_tokens > 1:
            df = df[df["review_length"] >= self.config.min_tokens]
        return df

