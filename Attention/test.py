from training import *
from model import *
from dataset import *
from config import *

import torch

if __name__ == "__main__":
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the dataset
    input_lang, output_lang, train_loader, valid_loader, test_loader = dataset.get_dataloader(
        32)  # type: ignore

    # Initialize the encoder and decoder models
    encoder = model.Encoder(
        type_=config.TYPE,
        num_layers_=config.ENCODER_NUM_LAYERS,
        hidden_dim_=config.HIDDEN_DIM,
        embed_dim_=config.EMBED_DIM,
        dropout_rate=config.DROPOUT_RATE,
        bidirectional_=config.BIDIRECTIONAL,
        batch_first_=True).to(device)
    decoder = model.AttentionDecoder(
        type_=config.TYPE,
        num_layers_=config.DECODER_NUM_LAYERS,
        hidden_dim_=config.HIDDEN_DIM,
        embed_dim_=config.EMBED_DIM,
        dropout_rate_=config.DROPOUT_RATE,
        bidirectional_=config.BIDIRECTIONAL,
        batch_first_=True).to(device)

    encoder.load_state_dict(torch.load("models/encoder_attention.pth"))
    decoder.load_state_dict(torch.load("models/decoder_attention.pth"))
    loss, acc = validate_one_step(encoder_model=encoder, decoder_model=decoder,
                            dataloader=test_loader, criterion=torch.nn.CrossEntropyLoss(), teacher_ratio=0)
    print(f"Loss: {loss}, Accuracy: {acc}")
    evaluate_dataset(encoder=encoder, decoder=decoder, dataloader=test_loader,
                     input_lang=input_lang, output_lang=output_lang, name='test_results')