import torch
import numpy as np
import pandas as pd
from typing import *
import random
import config
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
        self.index_to_word: Dict[int, str] = {0: 'SOS', 1: 'EOS'}
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


def data(language: str = 'ben', type: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the dataset for the specified language and type.

    Args:
        language: The language of the dataset. Defaults to 'ben' (Bengali).
        type: The type of the dataset. Must be one of 'train', 'val', or 'test'. Defaults to 'train'.

    Returns:
        The input and output arrays for the dataset.
    """
    path = "./aksharantar_sampled/{}/{}_{}.csv".format(
        language, language, type)
    df = pd.read_csv(path, header=None)
    return np.array(df[0]), np.array(df[1])


def set_words(
    lang: str,
    type: str,
) -> Tuple[Corpus, Corpus, List[List[str]]]:
    """
    Returns:
        A tuple containing the input and output languages and the word pairs.

    Args:
        lang: The language of the dataset.
    """
    input_lang, output_lang = Corpus('eng'), Corpus(lang)
    input_words, output_words = data(lang, type)
    word_pairs = [[input_words[i], output_words[i]]
                  for i in range(len(input_words))]
    for word in input_words:
        input_lang.add_word(word)
    for word in output_words:
        output_lang.add_word(word)
    return input_lang, output_lang, word_pairs


def word_to_index(lang: 'Corpus', word: 'str') -> List[int]:
    """
    Returns the indexes of the characters in the given word in the language.

    Args:
        lang: The language in which the word is written.
        word: The word to get the indexes of.

    Returns:
        The indexes of the characters in the word.
    """
    return [lang.word_to_index[char] for char in word]


def tensor_word(lang: 'Corpus', word: 'str') -> torch.Tensor:
    """
    Returns a tensor representing the given word in the given language.

    Args:
        lang: The language in which the word is written.
        word: The word to get the tensor of.

    Returns:
        A tensor representing the word. The tensor is a vector of size
        (len(word) + 1) where the final element is the end-of-sequence
        character.
    """
    indexes = word_to_index(lang, word)
    indexes.append(end)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_pair(input_lang: Corpus,
                 output_lang: Corpus,
                 pair: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the tensors representing the input and target words.

    Args:
        input_lang: The input language.
        output_lang: The output language.
        pair: The list of input and target words.

    Returns:
        A tuple containing the input and target tensors.
    """
    input_tensor = tensor_word(input_lang, pair[0])
    target_tensor = tensor_word(output_lang, pair[1])
    return (input_tensor, target_tensor)


def get_dataloader(batch_size: int = config.BATCH_SIZE) -> Tuple[Corpus, Corpus, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Returns the input and output languages and DataLoaders containing the
    training and validation pairs.

    Args:
        batch_size: The batch size of the DataLoaders.

    Returns:
        A tuple containing the input and output languages, and the DataLoaders.
        The first DataLoader is for training, and the second is for validation.
    """
    input_lang, output_lang, pairs = set_words('ben','train')
    n = len(pairs)
    input_ids = np.ones((n, config.MAX_LENGTH), dtype=np.int32)
    target_ids = np.ones((n, config.MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = word_to_index(input_lang, inp)
        tgt_ids = word_to_index(output_lang, tgt)
        inp_ids.append(end)
        tgt_ids.append(end)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = torch.utils.data.TensorDataset(torch.LongTensor(input_ids).to(device),
                                                torch.LongTensor(target_ids).to(device))

    train_sampler = torch.utils.data.RandomSampler(train_data)
    train_dataloader = torch.utils.data.DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size)
    
    
    _, _, pairs = set_words('ben','valid')
    n = len(pairs)
    input_ids = np.ones((n, config.MAX_LENGTH), dtype=np.int32)
    target_ids = np.ones((n, config.MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = word_to_index(input_lang, inp)
        tgt_ids = word_to_index(output_lang, tgt)
        inp_ids.append(end)
        tgt_ids.append(end)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    valid_data = torch.utils.data.TensorDataset(torch.LongTensor(input_ids).to(device),
                                                torch.LongTensor(target_ids).to(device))

    valid_sampler = torch.utils.data.RandomSampler(valid_data)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_data, sampler=valid_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader, valid_dataloader


if __name__ == '__main__':
    input_lang, output_lang, pairs = set_words('ben')
    print(random.choice(pairs))
    print("Number of words in input language: ", len(pairs))
    print("Number of characters in input language: ", input_lang.n_letters)
    print("Number of characters in output language: ", output_lang.n_letters)
    training_pairs = [tensors_pair(
        input_lang, output_lang, pair) for pair in pairs]
    print("Training pairs size: ", len(training_pairs))
