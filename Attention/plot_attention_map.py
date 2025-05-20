import os
import wandb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
from typing import *
import torch
import dataset
import model
import config
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_word(source: torch.Tensor,
             pred: torch.Tensor,
             input_lang: dataset.Corpus,
             output_lang: dataset.Corpus) -> Tuple[List[str], List[str]]:
    """
    Converts sequences of letter indices from source and predicted tensors into lists of words using the provided input and output language Corpus objects.

    Args:
        source (torch.Tensor): Tensor containing source word indices.
        pred (torch.Tensor): Tensor containing predicted word indices.
        input_lang (dataset.Corpus): Corpus object for the input language.
        output_lang (dataset.Corpus): Corpus object for the output language.

    Returns:
        Tuple[List[str], List[str]]: Lists of decoded source and predicted words.
    """
    source_words = []
    pred_words = []

    for word in source.cpu().numpy():
        temp_word = []
        for letter in word:
            if letter == 1:
                break
            temp_word.append(input_lang.index_to_word.get(letter, ''))
        source_words.append(''.join(temp_word))

    for word in pred.cpu().numpy():
        temp_word = []
        for letter in word:
            if letter == 1:
                break
            temp_word.append(output_lang.index_to_word.get(letter, ''))
        pred_words.append(''.join(temp_word))

    return source_words, pred_words


def get_attention_map(encoder: torch.nn.Module,
                      decoder: torch.nn.Module,
                      dataloader: torch.utils.data.DataLoader,
                      input_lang: dataset.Corpus,
                      output_lang: dataset.Corpus
                      ) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Generates attention maps for a batch of data using the provided encoder and decoder models.

    Args:
        encoder (torch.nn.Module): The encoder model.
        decoder (torch.nn.Module): The decoder model.
        dataloader (torch.utils.data.DataLoader): DataLoader providing input and target tensors.
        input_lang (dataset.Corpus): Corpus object for the input language.
        output_lang (dataset.Corpus): Corpus object for the output language.

    Returns:
        Tuple[List[str], List[str], np.ndarray]: A tuple containing lists of source and predicted words, and the attention map as a NumPy array.
    """
    for data in dataloader:
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        encoder_outputs, encoder_hidden = encoder(inputs)
        decoder_outputs, decoder_hidden, attention_map = decoder(
            encoder_outputs, encoder_hidden)
        outputs = decoder_outputs.argmax(-1)

        source, pred = get_word(inputs, outputs, input_lang, output_lang)
        break

    return source, pred, attention_map.detach().cpu().numpy()



def plot_attention(input_word: List[str],
                   predicted_word: List[str],
                   attention_map: np.ndarray,
                   file_name: str = None) -> None:
    """
    Plots the attention map for a single batch of data.

    Args:
        input_word: The input word as a list of strings.
        predicted_word: The predicted word as a list of strings.
        attention_map: The attention map as a numpy array.
        file_name: The file name to save the plot to.
    """
    os.makedirs('Attention_Heatmap', exist_ok=True)
    path = os.path.join(os.getcwd(), 'Attention_Heatmap', file_name)
    prop = fm.FontProperties(fname=os.path.join(
        os.getcwd(), 'Attention_Heatmap', 'Kalpurush.ttf'))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    img = ax.matshow(attention_map)

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + list(input_word), fontdict=fontdict, rotation=0)
    ax.set_yticklabels([''] + list(predicted_word),
                       fontdict=fontdict, fontproperties=prop)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlabel('Source (input)', fontsize=14)
    ax.set_ylabel('Prediction (output)', fontsize=14)
    ax.set_title(f'Attention heatmap', fontsize=16)

    cbar = fig.colorbar(img, ax=ax, location="right", pad=0.15)
    cbar.ax.set_ylabel("Attention weight", fontsize=12)
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.yaxis.set_ticks_position("left")


    plt.savefig(path)
    # with wandb.init(project="CS23E001_DL_3", name=file_name):
    #     wandb.log({'Attention Heatmap': wandb.Image(plt)})
    #     # plt.show()
    # wandb.finish()
    plt.close(fig)
    
def plot_attention_grid(source_words: List[List[str]],
                        predicted_words: List[List[str]],
                        attention_maps: List[np.ndarray],
                        file_name: str = "attention_grid.png") -> None:
    """
    Plots a 3x3 grid of attention maps.

    Args:
        source_words: List of 9 input words (each as a list of characters).
        predicted_words: List of 9 predicted words (each as a list of characters).
        attention_maps: List of 9 attention maps (each as 2D numpy array).
        file_name: Name of the file to save the combined image.
    """
    os.makedirs('Attention_Heatmap', exist_ok=True)
    path = os.path.join(os.getcwd(), 'Attention_Heatmap', file_name)

    try:
        prop = fm.FontProperties(fname=os.path.join('Attention_Heatmap', 'Kalpurush.ttf'))
    except:
        prop = None

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    fontdict = {'fontsize': 14}
    for i in range(9):
        ax = axes[i]
        img = ax.matshow(attention_maps[i], cmap='viridis')
        
        ax.set_xticklabels([''] + list(source_words[i]), fontdict=fontdict, rotation=0)
        ax.set_yticklabels([''] + list(predicted_words[i]),
                        fontdict=fontdict, fontproperties=prop)
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.set_xlabel('Source (input)', fontsize=14)
        ax.set_ylabel('Prediction (output)', fontsize=14)
        ax.set_title(f'Attention heatmap', fontsize=16)

        cbar = fig.colorbar(img, ax=ax, location="right", pad=0.15)
        cbar.ax.set_ylabel("Attention weight", fontsize=12)
        cbar.ax.yaxis.set_label_position("left")
        cbar.ax.yaxis.set_ticks_position("left")

    plt.tight_layout()
    plt.savefig(path)
    with wandb.init(project="DA24D402_DL_3", name=file_name):
        # wandb.log({'Attention Heatmap': wandb.Image(plt)})
        wandb.log({'Attention Grid': wandb.Image('Attention_Heatmap/heatmap_grid.png')})
        # plt.show()
    wandb.finish()
    # wandb.log({'Attention Grid': wandb.Image(path)})
    plt.close(fig)



if __name__ == "__main__":
    input_lang, output_lang, train_loader, valid_loader, test_loader = dataset.get_dataloader()

    encoder1 = model.Encoder().to(device)
    decoder1 = model.AttentionDecoder().to(device)

    encoder1.load_state_dict(torch.load('models/encoder_attention.pth'))
    decoder1.load_state_dict(torch.load('models/decoder_attention.pth'))

    source_words, pred_words, attentions = get_attention_map(
        encoder1, decoder1, test_loader, input_lang, output_lang)

    # for i in range(10):
    #     plot_attention(source_words[i], pred_words[i], attentions[i][1: len(pred_words[i]), 1: len(source_words[i])], f'heatmap_{i+1}.png')

    grid_sources = [list(source_words[i]) for i in range(9)]
    grid_preds = [list(pred_words[i]) for i in range(9)]
    grid_attns = [attentions[i][1: len(pred_words[i]), 1: len(source_words[i])] for i in range(9)]

    plot_attention_grid(grid_sources, grid_preds, grid_attns, "heatmap_grid.png")
