import torch
import model
import dataset
import config
import time
import tqdm
from typing import *
import os
import wandb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_step(
        encoder_model: 'model.Encoder',  # type: model.Model
        decoder_model: 'model.AttentionDecoder',
        encoder_optimizer: 'torch.optim.Optimizer',
        decoder_optimizer: 'torch.optim.Optimizer',
        criterion: 'torch.nn.CrossEntropyLoss',
        dataloader: 'torch.utils.data.DataLoader',
        teacher_ratio: float = 0.5,
) -> Tuple[float, float]:
    """
    Performs one step of training on the given data loader.

    Args:
        encoder_model: The encoder model to update.
        decoder_model: The decoder model to update.
        encoder_optimizer: The optimizer to use for the encoder model.
        decoder_optimizer: The optimizer to use for the decoder model.
        criterion: The loss function to use for computing the loss.
        teacher_ratio: The probability of using the target sequence instead of the decoder's generated sequence.
        dataloader: The data loader to use for training.

    Returns:
        The average loss and accuracy for this step.
    """
    encoder_model.train()
    decoder_model.train()

    losses = []
    accuracies = []
    for data in dataloader:
        input, output = data
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input = input.to(device)
        target = output.to(device)
        encoder_outputs, encoder_hidden = encoder_model(input)
        decoder_outputs, _, attention_weights = decoder_model(
            encoder_outputs, encoder_hidden, target, teacher_ratio)
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target.view(-1)
        )
        # find the accuracy among decoder outputs and target
        char_accuracy = ((decoder_outputs.argmax(-1) == target).sum() /
                         (decoder_outputs.size(0))*config.MAX_LENGTH)
        word_accuracy = ((decoder_outputs.argmax(-1) == target).all(1).sum() /
                         decoder_outputs.size(0)).item()

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        losses.append(loss.item())
        accuracies.append(word_accuracy)

    return np.mean(losses), np.mean(accuracies)


def validate_one_step(
        encoder_model: 'model.Encoder',  # type: model.Model
        decoder_model: 'model.AttentionDecoder',
        criterion: 'torch.nn.CrossEntropyLoss',
        dataloader: 'torch.utils.data.DataLoader',
        teacher_ratio: float = 0.0,
) -> Tuple[float, float]:
    """
    Performs one step of validation on the given data loader.

    Args:
        encoder_model: The encoder model to update.
        decoder_model: The decoder model to update.
        criterion: The loss function to use for computing the loss.
        teacher_ratio: The probability of using the target sequence instead of the decoder's generated sequence.
        dataloader: The data loader to use for training.

    Returns:
        The average loss and accuracy for this step.
    """
    encoder_model.eval()
    decoder_model.eval()

    losses = []
    accuracies = []
    with torch.no_grad():
        for data in dataloader:
            input, output = data

            input = input.to(device)
            target = output.to(device)
            encoder_outputs, encoder_hidden = encoder_model(input)
            decoder_outputs, _, _ = decoder_model(
                encoder_outputs, encoder_hidden, target, teacher_ratio)
            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target.view(-1)
            )
            # find the accuracy among decoder outputs and target
            char_accuracy = ((decoder_outputs.argmax(-1) == target).sum() /
                             (decoder_outputs.size(0))*config.MAX_LENGTH)
            word_accuracy = ((decoder_outputs.argmax(-1) == target).all(1).sum() /
                             decoder_outputs.size(0)).item()

            losses.append(loss.item())
            accuracies.append(word_accuracy)
    return np.mean(losses), np.mean(accuracies)


def evaluate_dataset(
        encoder: 'model.Encoder',  # type: model.Model
        decoder: 'model.AttentionDecoder',
        dataloader: 'torch.utils.data.DataLoader',  # type: torch.utils.data.DataLoader
        input_lang: 'dataset.Language',  # type: dataset.Language
        output_lang: 'dataset.Language',  # type: dataset.Language
        name: str = '',
        gen_heatmap: bool = False,  # type: bool
) -> List[List[str]]:
    """
    Evaluates the given model on the given dataset and returns the translations
    for a random sample of words in the given language pair.

    Args:
        encoder: The encoder model to use for encoding the input.
        decoder: The decoder model to use for generating the output.
        dataloader: The data loader to use for evaluating the model.
        input_lang: The input language.
        output_lang: The output language.
        gen_heatmap: Whether to generate heatmap or not.
        name: The name of the file to save the heatmap to.

    Returns:
        A list of lists of strings, where each inner list is a sequence of words
        in the output language corresponding to a sequence of words in the input
        language.
    """
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        datas = []
        for ind, data in enumerate(dataloader):
            input, output = data
            input = input.to(device)
            encoder_outputs, encoder_hidden = encoder(input)
            decoder_outputs, hidden_states, attention_weights = decoder(
                encoder_outputs, encoder_hidden)

            _, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()  # type: torch.Tensor

            for i in range(input.shape[0]):
                input_word = []
                predicted_word = []
                output_word = []

                for j in range(config.MAX_LENGTH):
                    if input[i][j].item() == dataset.end:
                        break
                    input_word.append(
                        input_lang.index_to_word[input[i][j].item()])
                for j in range(config.MAX_LENGTH):
                    if decoded_ids[i][j].item() == dataset.end:
                        break
                    predicted_word.append(
                        output_lang.index_to_word[decoded_ids[i][j].item()])
                for j in range(config.MAX_LENGTH):
                    if output[i][j].item() == dataset.end:
                        break
                    output_word.append(
                        output_lang.index_to_word[output[i][j].item()])

                if gen_heatmap:
                    length_input = len(input_word)
                    length_output = len(output_word)
                    attention_weight = attention_weights[i, :length_output, :length_input].cpu().detach().numpy()

                    sns.heatmap(attention_weight, cmap="viridis")
                    plt.xticks(ticks=range(length_input), labels=input_word, rotation=90)
                    plt.yticks(ticks=range(length_output), labels=output_word, rotation=0)
                    plt.xlabel("Input")
                    plt.ylabel("Output")

                    os.makedirs("samples", exist_ok=True)
                    plt.tight_layout()
                    plt.savefig(f'samples/heatmap_{name}_{ind}.png')
                    plt.close()
                input_word = ''.join(input_word)
                predicted_word = ''.join(predicted_word)
                output_word = ''.join(output_word)
                datas.append([input_word, predicted_word, output_word])
    with open('samples/' + name + '.txt', 'w') as f:
        for item in datas:
            for datum in item:
                f.write("%s,\t" % datum)
            f.write("\n")
    return datas


def evaluate(
        encoder: 'model.Encoder',  # type: model.Model
        decoder: 'model.AttentionDecoder',
        word: str,  # type: str
        input_lang: 'dataset.Language',  # type: dataset.Language
        output_lang: 'dataset.Language',  # type: dataset.Language
) -> List[str]:
    """
    Evaluates the given encoder and decoder models on the given word.

    Args:
        encoder: The encoder model to update.
        decoder: The decoder model to update.
        word: The word to translate from the input language to the output language.
        input_lang: The input language.
        output_lang: The output language.

    Returns:
        A list of the decoded words.
    """
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        input_tensor = dataset.tensor_word(input_lang, word).reshape(1, -1)
        input_tensor = torch.nn.ConstantPad1d(
            (0, config.MAX_LENGTH-input_tensor.shape[1]), 0)(input_tensor)
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        
        decoder_outputs, decoder_hidden, decoder_attention = decoder(
            encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == dataset.end:
                break
            decoded_words.append(output_lang.index_to_word[idx.item()])
    return decoded_words


def train(train_loader: torch.utils.data.DataLoader,  # type: torch.utils.data.DataLoader
          valid_loader: torch.utils.data.DataLoader,  # type: torch.utils.data.DataLoader
          test_loader: torch.utils.data.DataLoader,  # type: torch.utils.data.DataLoader
          input_lang: 'dataset.Language',  # type: dataset.Language
          output_lang: 'dataset.Language',  # type: dataset.Language
          encoder: 'model.Encoder',  # type: model.Encoder
          decoder: 'model.AttentionDecoder',  # type: model.Decoder
          epochs: int,
          wandb_log: bool = False,
          name: str = '',
          learning_rate: float = config.LEARNING_RATE
          ) -> None:
    """
    Trains the given encoder and decoder models on the given data loaders.

    Args:
        train_loader: The data loader for the training data.
        valid_loader: The data loader for the validation data.
        encoder: The encoder model to update.
        decoder: The decoder model to update.
        epochs: The number of epochs to train for.
        wandb_log: Whether to log to Weights & Biases.
        learning_rate: The learning rate for the Adam optimizers.

    Returns:
        None.
    """
    start_time = time.time()
    losses = []  # type: List[float]
    total_loss = 0.0  # type: float

    encoder_optimizer = torch.optim.Adam(
        encoder.parameters(), lr=learning_rate, weight_decay=1e-5)
    decoder_optimizer = torch.optim.Adam(
        decoder.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    encoder_scheduler = torch.optim.lr_scheduler.LinearLR(
        encoder_optimizer, start_factor=1, end_factor=0.5, total_iters=epochs)
    decoder_scheduler = torch.optim.lr_scheduler.LinearLR(
        decoder_optimizer, start_factor=1, end_factor=0.5, total_iters=epochs)
    bar = tqdm.tqdm(range(0, epochs+1))
    for epoch in bar:
        train_loss, train_acc = train_one_step(
            encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, train_loader)
        bar.set_description(
            f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')

        valid_loss, valid_acc = validate_one_step(
            encoder, decoder, criterion, valid_loader)
        bar.set_description(
            f'Valid Loss: {valid_loss:.4f} Acc: {valid_acc:.4f}')
        # evaluate_dataset(encoder, decoder, valid_loader,
        #                                 input_lang, output_lang,True, name)

        if wandb_log:
            wandb.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'valid_loss': valid_loss,
                'valid_acc': valid_acc
            })
        if test_loader is not None:
            test_loss, test_acc = validate_one_step(
                encoder, decoder, criterion, test_loader)
            if wandb_log:
                wandb.log({
                    'test_loss': test_loss,
                    'test_acc': test_acc
                })
        if epoch % config.PRINT_EVERY == 0:
            print(
                f'Time elapsed: {(time.time() - start_time) / 60:.2f} min')
            print('Epoch [{}/{}],Train      Loss: {:.4f} Accuracy: {}'.format(epoch,
                                                                              epochs, train_loss, train_acc))
            print('Epoch [{}/{}],Validation Loss: {:.4f} Accuracy: {}'.format(epoch,
                                                                              epochs, valid_loss, valid_acc))
            print()
            if test_loader is None:
                evaluate_dataset(encoder, decoder, valid_loader,
                                 input_lang, output_lang, name)
            else:
                print(
                    f'Test      Loss: {test_loss:.4f} Accuracy: {test_acc:.4f}')
                evaluate_dataset(encoder, decoder, test_loader,
                                 input_lang, output_lang, name='result_best')
        encoder_scheduler.step()
        decoder_scheduler.step()
        
        torch.save(encoder.state_dict(), 'models/encoder_' + name + '.pth')
        torch.save(decoder.state_dict(), 'models/decoder_' + name + '.pth')


if __name__ == '__main__':
    input_lang, output_lang, train_loader, valid_loader, test_loader = dataset.get_dataloader()

    
    train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        input_lang=input_lang,
        output_lang=output_lang,
        encoder=model.Encoder(
            type_=config.TYPE,
            num_layers_=config.ENCODER_NUM_LAYERS,
            hidden_dim_=config.HIDDEN_DIM,
            embed_dim_=config.EMBED_DIM,
            dropout_rate=config.DROPOUT_RATE,
            bidirectional_=config.BIDIRECTIONAL,
            batch_first_=True).to(device),
        decoder=model.AttentionDecoder(
            type_=config.TYPE,
            num_layers_=config.DECODER_NUM_LAYERS,
            hidden_dim_=config.HIDDEN_DIM,
            embed_dim_=config.EMBED_DIM,
            dropout_rate_=config.DROPOUT_RATE,
            bidirectional_=config.BIDIRECTIONAL,
            batch_first_=True).to(device),
        epochs=config.EPOCHS, wandb_log=False, name='attention', learning_rate=config.LEARNING_RATE
    )
    
