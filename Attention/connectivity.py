import torch
from model import *
from config import *
from dataset import *
import json
import wandb
import numpy as np
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def export_attention_visualization_multi(encoder,decoder, input_tensors, input_lang, output_lang, path="web_page/attention_data.json"):
    encoder.eval()
    decoder.eval()
    export_list = []
    with torch.no_grad():
        for input_tensor in input_tensors:
            encoder_outputs, encoder_hidden = encoder(input_tensor.unsqueeze(0).to(DEVICE))
            decoder_outputs, _, attentions = decoder(encoder_outputs, encoder_hidden)
            _, predictions = decoder_outputs.topk(1)
            predictions = predictions.squeeze(0)

            input_chars = [input_lang.index_to_word[idx.item()] for idx in input_tensor if idx.item() != end]
            output_chars = []
            for idx in predictions:
                if idx.item() == end:
                    break
                output_chars.append(output_lang.index_to_word[idx.item()])

            attention_matrix = attentions.squeeze().cpu().numpy()
            attention_matrix = np.maximum(attention_matrix, 0)
            if attention_matrix.max() > 0:
                attention_matrix = attention_matrix / attention_matrix.max()
            data = {
                "input": input_chars,
                "prediction": output_chars,
                "attention": attention_matrix.tolist()
            }
            export_list.append(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(export_list, f, ensure_ascii=False, indent=2)
    print(f"Exported {len(export_list)} samples to {path}")

if __name__ == '__main__':
    input_lang, output_lang, train_loader, valid_loader, test_loader = dataset.get_dataloader(
        32)  # type: ignore

    # Initialize the encoder and decoder models
    encoder = Encoder(
        type_=config.TYPE,
        num_layers_=config.ENCODER_NUM_LAYERS,
        hidden_dim_=config.HIDDEN_DIM,
        embed_dim_=config.EMBED_DIM,
        dropout_rate=config.DROPOUT_RATE,
        bidirectional_=config.BIDIRECTIONAL,
        batch_first_=True).to(device)
    decoder = AttentionDecoder(
        type_=config.TYPE,
        num_layers_=config.DECODER_NUM_LAYERS,
        hidden_dim_=config.HIDDEN_DIM,
        embed_dim_=config.EMBED_DIM,
        dropout_rate_=config.DROPOUT_RATE,
        bidirectional_=config.BIDIRECTIONAL,
        batch_first_=True).to(device)

    encoder.load_state_dict(torch.load("models/encoder_attention.pth"))
    decoder.load_state_dict(torch.load("models/decoder_attention.pth"))
    
    batch = next(iter(test_loader))
    input_batch, _ = batch
    indices = random.sample(range(input_batch.shape[0]), 10)
    input_tensors = [input_batch[idx] for idx in indices]
    export_attention_visualization_multi(encoder, decoder, input_tensors, input_lang, output_lang, path="web_page/attention_data.json")

    with open("web_page/attention_data.json", "r", encoding="utf-8") as f:
        attention_data = json.load(f)

    # Inject JSON as string
    with open("web_page/static.html", "r", encoding="utf-8") as f:
        template = f.read()

    html_content = template.replace("REPLACE_THIS_WITH_JSON", json.dumps(attention_data))

    with open("web_page/attention.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    # # Log to wandb
    with wandb.init(project="DA24D402_DL_3", name='attention visualization'):
        wandb.log({"Attention Visualization": wandb.Html("web_page/attention.html")})
    
