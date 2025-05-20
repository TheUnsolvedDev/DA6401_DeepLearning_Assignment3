import torch
import numpy as np
import pandas as pd
from typing import *
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start = 0
end = 1

class Corpus:
    def __init__(self, language: str = 'beng') -> None:
        """Initializes the corpus.

        Args:
            language: The language of the corpus. Defaults to 'beng' (Bengali).
        """
        self.language = language
        self.word_to_index: Dict[str, int] = {}
        self.word_to_count: Dict[str, int] = {}
        self.index_to_word: Dict[int, str] = {}
        self.n_letters: int = 2

    def add_word(self, word: str) -> None:
        """Adds a word to the corpus.

        Args:
            word: The word to add.
        """
        for letter in word:
            self.add_letter(letter)

    def add_letter(self, letter: str) -> None:
        """Adds a letter to the corpus.

        Args:
            letter: The letter to add.
        """
        if letter in self.word_to_index:
            self.word_to_count[letter] += 1
        else:
            self.word_to_index[letter] = self.n_letters
            self.word_to_count[letter] = 1
            self.index_to_word[self.n_letters] = letter
            self.n_letters += 1
            
class Dataset(torch.nn.utils.Dataset):
    def __init__(self, corpus: Corpus, start: int = 0, end: int = 1) -> None:
        """Initializes the dataset.

        Args:
            corpus: The corpus to use.
            start: The index of the first word to use. Defaults to 0.
            end: The index of the last word to use. Defaults to 1.
        """
        self.corpus = corpus
        self.start = start
        self.end = end
        self.data = self._build_dataset()   
        
    def __len__(self) -> int:
        """Returns the number of words in the dataset.

        Returns:
            The number of words in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the word and its target word.

        Args:
            index: The index of the word to get.

        Returns:
            The word and its target word.
        """
        return self.data[index]    

    def _build_dataset(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Builds the dataset.

        Returns:
            The dataset.
        """
        dataset = []
        for i in range(self.start, self.end):
            word = tensor_word(self.corpus, self.data[i][0])
            target = tensor_word(self.corpus, self.data[i][1])
            dataset.append((word, target))
        return dataset  
    