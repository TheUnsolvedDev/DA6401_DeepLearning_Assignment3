import torch
import config
import dataset
import random
from typing import *
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(True)


def cell(cell_type: str) -> Type[torch.nn.RNNBase]:
    """
    Returns the appropriate RNN class based on the cell_type string.

    Args:
        cell_type (str): The type of RNN cell to return. Must be one of 'LSTM', 'GRU', or 'RNN'.

    Returns:
        Type[torch.nn.RNNBase]: The desired RNN class.

    Raises:
        Exception: If an invalid cell_type is given.
    """
    if cell_type == 'LSTM':
        return torch.nn.LSTM
    elif cell_type == 'GRU':
        return torch.nn.GRU
    elif cell_type == 'RNN':
        return torch.nn.RNN
    else:
        raise Exception("Invalid cell type")


class Encoder(torch.nn.Module):
    def __init__(self,
                 # The type of RNN cell to use. Must be one of 'LSTM', 'GRU', or 'RNN'.
                 type_: str = config.TYPE,
                 # The number of layers in the RNN.
                 num_layers_: int = config.ENCODER_NUM_LAYERS,
                 # The number of features in the hidden state.
                 hidden_dim_: int = config.HIDDEN_DIM,
                 # The number of features in the embedded input.
                 embed_dim_: int = config.EMBED_DIM,
                 # The number of features in the input.
                 input_dim_: int = config.INPUT_DIM,
                 # The dropout rate to use in the EncoderRNN.
                 dropout_rate: float = config.DROPOUT_RATE,
                 # If True, use a bidirectional RNN.
                 bidirectional_: bool = config.BIDIRECTIONAL,
                 # If True, the input and output tensors are provided as (batch, seq, feature).
                 batch_first_: bool = True
                 ) -> None:
        """
        Initializes the EncoderRNN.

        Args:
            type_: The type of RNN cell to use. Must be one of 'LSTM', 'GRU', or 'RNN'.
            num_layers_: The number of layers in the RNN.
            hidden_dim_: The number of features in the hidden state.
            input_dim_: The number of features in the input.
            embed_dim_: The number of features in the embedded input.
            dropout_rate: The dropout rate to use in the EncoderRNN.
            bidirectional_: If True, use a bidirectional RNN.
            batch_first_: If True, the input and output tensors are provided as (batch, seq, feature).

        Returns:
            None
        """
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim_
        self.type = type_
        self.num_layers = num_layers_
        self.batch_first = batch_first_
        self.embed_dim = embed_dim_
        self.input_dim = input_dim_
        self.dropout_rate = 0 if num_layers_ <= 1 else dropout_rate
        self.bidirectional = bidirectional_

        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.embedding = torch.nn.Embedding(  # type: ignore
            self.input_dim, self.embed_dim)
        self.cell: torch.nn.RNNBase = cell(type_)(  # type: ignore
            self.embed_dim, self.hidden_dim, num_layers=num_layers_,  # type: ignore
            batch_first=batch_first_, dropout=self.dropout_rate, bidirectional=bidirectional_)

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the Encoder.

        Args:
            input_tensor: A tensor of shape (batch_size, max_length) containing the input to the Encoder.

        Returns:
            A tuple of the output and the final hidden state of the Encoder. The output is a tensor of shape (batch_size, max_length, hidden_dim) and the final hidden state is a tuple of two tensors of shape (num_layers*(1+bidirectional), batch_size, hidden_dim).
        """
        # Initialize the hidden state of the LSTM layers to zero tensors with the correct shape
        encoder_hidden = torch.zeros(
            self.num_layers*(1+self.bidirectional), input_tensor.size(0), self.hidden_dim, device=device)  # type: ignore
        encoder_cell = torch.zeros(
            self.num_layers*(1+self.bidirectional), input_tensor.size(0), self.hidden_dim, device=device)  # type: ignore
        # encoder_outputs = []

        if self.type == 'LSTM':
            encoder_outputs, (encoder_hidden, encoder_cell) = self.forward_step(
                input_tensor, (encoder_hidden, encoder_cell))
        else:
            encoder_outputs, encoder_hidden = self.forward_step(
                input_tensor, encoder_hidden)

        # Return the concatenated output tensor and the final hidden state of the EncoderRNN
        if self.type == 'LSTM':
            encoder_hidden = (encoder_hidden, encoder_cell)
        return encoder_outputs, encoder_hidden

    def forward_step(
            self,
            # A tensor of shape (1, 1) containing the input to the RNN.
            input_: torch.Tensor,  # type: torch.Tensor
            hidden: torch.Tensor,  # The initial hidden state of the RNN.
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the EncoderRNN.

        Args:
            input_: A tensor of shape (1, 1) containing the input to the RNN.
            hidden: The initial hidden state of the RNN.

        Returns:
            A tuple of the output and the final hidden state of the RNN. The final hidden state is a tuple of the hidden state and cell state of the LSTM layers.
        """
        # Embed the input and apply dropout and a ReLU activation function
        embedded = self.embedding(input_)
        embedded = self.dropout(embedded)

        # Perform a single forward pass through the RNN
        if self.type == 'LSTM':
            hidden_state, cell_state = hidden
            output, (hidden_state, cell_state) = self.cell(
                embedded, (hidden_state, cell_state))
            hidden_state = (hidden_state, cell_state)
        else:
            output, hidden_state = self.cell(embedded, hidden)
        return output, hidden_state


class BahdanauAttention(torch.nn.Module):
    def __init__(
            self,
            # The hidden size of the RNN. (int)
            num_layers: int = config.DECODER_NUM_LAYERS,  # type: int
            hidden_size: int = config.HIDDEN_DIM,  # type: int
            bidirectional: bool = config.BIDIRECTIONAL,  # type: bool
    ) -> None:
        """
        Initialize the BahdanauAttention module.

        Args:
            hidden_size: The hidden size of the RNN. (int)
        """
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.Wa = torch.nn.Linear(hidden_size*(1+bidirectional), hidden_size*(
            1+bidirectional), bias=False)  # type: torch.nn.Linear
        self.Ua = torch.nn.Linear(hidden_size*(1+bidirectional), hidden_size*(
            1+bidirectional), bias=False)  # type: torch.nn.Linear
        # type: torch.nn.Linear
        self.Va = torch.nn.Linear(hidden_size*(1+bidirectional), 1, bias=False)

    def forward(
            self,
            # type: torch.Tensor  # (batch_size, 1, hidden_size)
            query: torch.Tensor,
            # type: torch.Tensor  # (batch_size, sequence_length, hidden_size)
            keys: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:  
        """
        Calculate the attention weights and the weighted sum of the values.

        Args:
            query (torch.Tensor): The input sequence. (batch_size, 1, hidden_size)
            keys (torch.Tensor): The sequence of keys. (batch_size, sequence_length, hidden_size)

        Returns:
            A tuple of the weighted sum of the values and the attention weights.
        """

        if self.bidirectional:
            query_t = torch.cat([query[:,-2,:],query[:,-1,:]],dim=1).unsqueeze(1)
            
        else:
            query_t = query[:,-1,:].unsqueeze(1)            
        query_proj = self.Wa(query_t)
        
        keys_proj = self.Ua(keys)
        energy = torch.tanh(query_proj+keys_proj)

        attention_score = self.Va(energy).squeeze(2)
        attention_weights = torch.nn.functional.softmax(attention_score,dim=1).unsqueeze(1)
        context = torch.bmm(attention_weights,keys)
        return context,attention_weights


class AttentionDecoder(torch.nn.Module):
    def __init__(
            self,
            # The type of RNN cell to use. Must be one of 'LSTM', 'GRU', or 'RNN'. (str)
            type_: str = config.TYPE,  # type: str
            # The number of layers in the RNN. (int)
            num_layers_: int = config.DECODER_NUM_LAYERS,  # type: int
            # The hidden size of the RNN. (int)
            hidden_dim_: int = config.HIDDEN_DIM,  # type: int
            # If True, the input and output tensors are provided as (batch, seq, feature). (bool)
            dropout_rate_: float = config.DROPOUT_RATE,  # type: float
            # If True, the RNN is bidirectional. (bool)
            bidirectional_: bool = config.BIDIRECTIONAL,  # type: bool
            # If True, the input and output tensors are provided as (batch, seq, feature). (bool)
            batch_first_: bool = True,  # type: bool
            # The embedding dimension of the DecoderRNN. (int)
            embed_dim_: int = config.EMBED_DIM,  # type: int
            # The output dimension of the DecoderRNN. (int)
            output_dim_: int = config.OUTPUT_DIM,  # type: int
    ) -> None:
        """
        Initializes the DecoderRNN.

        Args:
            type_: The type of RNN cell to use. Must be one of 'LSTM', 'GRU', or 'RNN'.
            num_layers_: The number of layers in the RNN.
            hidden_dim_: The hidden size of the RNN.
            dropout_rate_: The dropout rate to use for the RNN.
            bidirectional_: If True, the RNN is bidirectional.
            batch_first_: If True, the input and output tensors are provided as (batch, seq, feature).
            output_dim_: The output dimension of the DecoderRNN.
            embed_dim_: The embedding dimension of the DecoderRNN.

        Returns:
            None
        """
        super(AttentionDecoder, self).__init__()
        self.type = type_
        self.num_layers = num_layers_
        self.hidden_dim = hidden_dim_
        self.batch_first = batch_first_
        self.output_dim = output_dim_
        self.embed_dim = embed_dim_
        self.dropout_rate = 0 if num_layers_ <= 1 else dropout_rate_
        self.bidirectional = bidirectional_

        self.attention = BahdanauAttention(
            self.num_layers, hidden_dim_, self.bidirectional)
        self.embedding = torch.nn.Embedding(  # type: torch.nn.Embedding
            self.output_dim, self.embed_dim)  # input_dim: int, embed_dim: int
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.type = type_  # type: str
        self.cell = cell(type_)(  # type: torch.nn.RNNBase
            self.embed_dim+self.hidden_dim*(1+self.bidirectional), self.hidden_dim, num_layers=num_layers_, batch_first=batch_first_, dropout=self.dropout_rate, bidirectional=bidirectional_)
        self.out = torch.nn.Linear(  # type: torch.nn.Linear
            self.hidden_dim*(1+self.bidirectional), self.output_dim)  # in_features: int, out_features: int

    def forward(
            self,
            # type: torch.Tensor  # (batch_size, seq_len, hidden_size)
            encoder_outputs: torch.Tensor,
            # type: torch.Tensor  # (num_layers * num_directions, batch_size, hidden_size)
            encoder_hidden: torch.Tensor,
            # type: Optional[torch.Tensor]  # (batch_size, seq_len)
            target_tensor: Optional[torch.Tensor] = None,
            # type: float
            teacher_ratio: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]: 
        """
        Forward pass through the DecoderRNN.

        Args:
            encoder_outputs: The output of the encoder.
            encoder_hidden: The final hidden state of the encoder.
            target_tensor: An optional tensor containing the target sequence.
            teacher_ratio: The probability of using the target sequence instead of the decoder's generated sequence.

        Returns:
            A tuple of the output and the final hidden state of the RNN.
        """
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(
            batch_size,
            1,
            dtype=torch.long,
            device=device).fill_(dataset.start)  # type: torch.Tensor  # (batch_size, 1)
        decoder_outputs = []
        attentions = []
        if self.type == 'LSTM':
            encoder_hidden, encoder_cell = encoder_hidden
            if encoder_hidden.shape[0] != self.num_layers*(1+self.bidirectional):
                encoder_hidden_reshaped = torch.stack(
                    [encoder_hidden.mean(0) for i in range(self.num_layers*(1+self.bidirectional))])
                encoder_cell_reshaped = torch.stack(
                    [encoder_cell.mean(0) for i in range(self.num_layers*(1+self.bidirectional))])
                decoder_cell = encoder_cell_reshaped
                decoder_hidden = encoder_hidden_reshaped
            else:
                decoder_cell = encoder_cell
                decoder_hidden = encoder_hidden

        else:
            if encoder_hidden.shape[0] != self.num_layers*(1+self.bidirectional):
                encoder_hidden_reshaped = torch.stack(
                    [encoder_hidden.mean(0) for i in range(self.num_layers*(1+self.bidirectional))])
                decoder_hidden = encoder_hidden_reshaped
            else:
                decoder_hidden = encoder_hidden

        for i in range(config.MAX_LENGTH):
            if self.type == 'LSTM':
                decoder_output, (decoder_hidden, decoder_cell), attn_weights = self.forward_step(
                    decoder_input, (decoder_hidden, decoder_cell), encoder_outputs)
            else:
                decoder_output, decoder_hidden, attn_weights = self.forward_step(
                    decoder_input, decoder_hidden, encoder_outputs
                )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None and teacher_ratio > random.random():
                decoder_input = target_tensor[:, i].unsqueeze(
                    1)  # type: torch.Tensor  # (batch_size, 1)  # Teacher forcing
            else:
                _, topi = decoder_output.topk(1)
                # type: torch.Tensor  # (batch_size, 1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        attentions = torch.cat(attentions, dim=1)
        return decoder_outputs, decoder_hidden, attentions

    def forward_step(
            self,
            input_: torch.Tensor,  # type: torch.Tensor  # (1, 1)
            hidden: torch.Tensor,  # type: torch.Tensor  # (num_layers * num_directions, batch_size, hidden_size)
            outputs: torch.Tensor,  # type: torch.Tensor  # (seq_len, batch_size, hidden_size)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        """
        Forward pass through the DecoderRNN.

        Args:
            input_ (torch.Tensor): A tensor of shape (1, 1) containing the input to the RNN.
            hidden (torch.Tensor): The initial hidden state of the RNN.
            outputs (torch.Tensor): The output of the encoder.

        Returns:
            A tuple of the output, the final hidden state of the RNN, and the attention weights.
        """
        embed = self.embedding(input_)
        active_embed = torch.nn.functional.relu(embed)
        if isinstance(self.cell, torch.nn.LSTM):
            hidden_state, cell_state = hidden
            query = hidden_state.permute(1, 0, 2)
            context, attn_weights = self.attention(query, outputs)
            active_embed = torch.cat((active_embed, context), dim=2)
            output, (hidden_state, cell_state) = self.cell(
                active_embed, (hidden_state, cell_state))
            output = self.dropout(output)
            output = self.out(output)
            return output, (hidden_state, cell_state), attn_weights
        else:
            query = hidden.permute(1, 0, 2)
            context, attn_weights = self.attention(query, outputs)
            active_embed = torch.cat((active_embed, context), dim=2)
            output, hidden_state = self.cell(active_embed, hidden)
            output = self.dropout(output)
            output = self.out(output)
            return output, hidden_state, attn_weights
