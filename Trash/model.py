import torch
import config
import dataset
import random
from typing import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
                 type_: str,  # The type of RNN cell to use. Must be one of 'LSTM', 'GRU', or 'RNN'.
                 num_layers_: int,  # The number of layers in the RNN.
                 hidden_dim_: int,  # The number of features in the hidden state.
                 input_dim_: int,  # The number of features in the input.
                 embed_dim_: int,  # The number of features in the embedded input.
                 dropout_rate: float = 0.1,  # The dropout rate to use in the EncoderRNN.
                 bidirectional_: bool = False,  # If True, use a bidirectional RNN.
                 batch_first_: bool = True  # If True, the input and output tensors are provided as (batch, seq, feature).
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
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional_

        self.embedding = torch.nn.Embedding(  # type: ignore
            self.input_dim, self.embed_dim)
        self.dropout: torch.nn.Dropout = torch.nn.Dropout(dropout_rate)
        self.cell: torch.nn.RNNBase = cell(type_)(  # type: ignore
            self.embed_dim, self.hidden_dim, num_layers=num_layers_,  # type: ignore
            batch_first=batch_first_, dropout=dropout_rate, bidirectional=bidirectional_)

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
        encoder_outputs = []

        for i in range(config.MAX_LENGTH):
            # Get the i-th element of the sequence
            encoder_input = input_tensor[:, i].reshape(-1, 1)

            # Perform a single forward pass through the EncoderRNN
            if self.type == 'LSTM':
                encoder_output, (encoder_hidden, encoder_cell) = self.forward_step(
                    encoder_input, (encoder_hidden, encoder_cell))
            else:
                encoder_output, encoder_hidden = self.forward_step(
                    encoder_input, encoder_hidden)

            # Append the output of the forward pass to the list of outputs
            encoder_outputs.append(encoder_output)

        # Concatenate all of the output tensors from the EncoderRNN into a single tensor
        encoder_outputs = torch.cat(encoder_outputs, dim=1)  # type: ignore

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
        embedded = torch.nn.functional.relu(embedded)

        # Perform a single forward pass through the RNN
        if isinstance(self.cell, torch.nn.LSTM):
            hidden_state, cell_state = hidden
            output, (hidden_state, cell_state) = self.cell(
                embedded, (hidden_state, cell_state))
        else:
            output, hidden_state = self.cell(embedded, hidden)

        return output, hidden_state


class Decoder(torch.nn.Module):
    def __init__(
            self,
            # The type of RNN cell to use. Must be one of 'LSTM', 'GRU', or 'RNN'. (str)
            type_: str,  # type: str
            # The number of layers in the RNN. (int)
            num_layers_: int,  # type: int
            # The hidden size of the RNN. (int)
            hidden_dim_: int,  # type: int
            # If True, the input and output tensors are provided as (batch, seq, feature). (bool)
            dropout_rate_: float,  # type: float
            # If True, the RNN is bidirectional. (bool)
            bidirectional_: bool,  # type: bool
            # If True, the input and output tensors are provided as (batch, seq, feature). (bool)
            batch_first_: bool,  # type: bool
            # The output dimension of the DecoderRNN. (int)
            output_dim_: int,  # type: int
            # The embedding dimension of the DecoderRNN. (int)
            embed_dim_: int,  # type: int
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
        super(Decoder, self).__init__()
        self.type = type_
        self.num_layers = num_layers_
        self.hidden_dim = hidden_dim_
        self.batch_first = batch_first_
        self.output_dim = output_dim_
        self.embed_dim = embed_dim_
        self.dropout_rate = dropout_rate_
        self.bidirectional = bidirectional_

        self.embedding = torch.nn.Embedding(  # type: torch.nn.Embedding
            self.output_dim, self.embed_dim)  # input_dim: int, embed_dim: int
        self.dropout = torch.nn.Dropout(dropout_rate_)  # type: torch.nn.Dropout
        self.type = type_  # type: str
        self.cell = cell(type_)(  # type: torch.nn.RNNBase
            self.embed_dim, self.hidden_dim, num_layers=num_layers_, batch_first=batch_first_, dropout=dropout_rate_, bidirectional=bidirectional_)
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # type: Tuple[torch.Tensor, torch.Tensor]  # (batch_size, seq_len, output_size), (num_layers * num_directions, batch_size, hidden_size)
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
        if self.type == 'LSTM':
            encoder_hidden, encoder_cell = encoder_hidden
            if encoder_hidden.shape[0] != self.num_layers:
                encoder_hidden = encoder_hidden.mean(0)
                encoder_hidden = torch.stack(
                    [encoder_hidden for i in range(self.num_layers*(1+self.bidirectional))])
                encoder_cell = encoder_cell.mean(0)
                encoder_cell = torch.stack(
                    [encoder_cell for i in range(self.num_layers*(1+self.bidirectional))])
            decoder_cell = encoder_cell
            decoder_hidden = encoder_hidden
        else:
            if encoder_hidden.shape[0] != self.num_layers:
                encoder_hidden = encoder_hidden.mean(0)
                encoder_hidden = torch.stack(
                    [encoder_hidden for i in range(self.num_layers*(1+self.bidirectional))])
            decoder_hidden = encoder_hidden
            
        for i in range(config.MAX_LENGTH):
            if self.type == 'LSTM':
                decoder_output, (decoder_hidden, decoder_cell) = self.forward_step(
                    decoder_input, (decoder_hidden, decoder_cell))
            else:
                decoder_output, decoder_hidden = self.forward_step(
                    decoder_input, decoder_hidden
                )
            decoder_outputs.append(decoder_output)

            if target_tensor is not None and teacher_ratio > random.random():
                decoder_input = target_tensor[:, i].unsqueeze(
                    1)  # type: torch.Tensor  # (batch_size, 1)  # Teacher forcing
            else:
                _, topi = decoder_output.topk(1)
                # type: torch.Tensor  # (batch_size, 1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs, decoder_hidden

    def forward_step(
        self,
        input_: torch.Tensor,  # type: torch.Tensor  # (1, 1)
        hidden: torch.Tensor,  # type: torch.Tensor  # (num_layers * num_directions, batch_size, hidden_size)
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # type: Tuple[torch.Tensor, torch.Tensor]  # (1, 1, output_size), (num_layers * num_directions, batch_size, hidden_size)
        """
        Forward pass through the DecoderRNN.

        Args:
            input_ (torch.Tensor): A tensor of shape (1, 1) containing the input to the RNN.
            hidden (torch.Tensor): The initial hidden state of the RNN.

        Returns:
            A tuple of the output and the final hidden state of the RNN.
        """
        embed = self.embedding(input_)
        embed = self.dropout(embed)
        active_embed = torch.nn.functional.relu(embed)
        if isinstance(self.cell, torch.nn.LSTM):
            hidden_state, cell_state = hidden
            output, (hidden_state, cell_state) = self.cell(
                active_embed, (hidden_state, cell_state))
            output = self.out(output)
            return output, (hidden_state, cell_state)
        else:
            output, hidden_state = self.cell(active_embed, hidden)
            output = self.out(output)
            return output, hidden_state
