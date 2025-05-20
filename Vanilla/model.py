import torch
import config
import dataset
import random
from typing import *
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    def __init__(
            self,
            # The type of RNN cell to use. Must be one of 'LSTM', 'GRU', or 'RNN'. (str)
            type_: str = config.TYPE,  # type: str
            # The number of layers in the RNN. (int)
            num_layers_: int = config.ENCODER_NUM_LAYERS,  # type: int
            # The number of features in the hidden state. (int)
            hidden_dim_: int = config.HIDDEN_DIM,  # type: int
            # The number of features in the embedded input. (int)
            embed_dim_: int = config.EMBED_DIM,  # type: int
            # The number of features in the input. (int)
            input_dim_: int = config.INPUT_DIM,  # type: int
            # The dropout rate to use in the EncoderRNN. (float)
            dropout_rate: float = config.DROPOUT_RATE,  # type: float
            # If True, use a bidirectional RNN. (bool)
            bidirectional_: bool = config.BIDIRECTIONAL,  # type: bool
            # If True, the input and output tensors are provided as (batch, seq, feature). (bool)
            batch_first_: bool = True) -> None:
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

        # for i in range(config.MAX_LENGTH):
        #     # Get the i-th element of the sequence
        #     encoder_input = input_tensor[:, i].reshape(-1, 1)

        #     # Perform a single forward pass through the EncoderRNN
        #     if self.type == 'LSTM':
        #         encoder_output, (encoder_hidden, encoder_cell) = self.forward_step(
        #             encoder_input, (encoder_hidden, encoder_cell))
        #     else:
        #         encoder_output, encoder_hidden = self.forward_step(
        #             encoder_input, encoder_hidden)

        #     # Append the output of the forward pass to the list of outputs
        #     encoder_outputs.append(encoder_output)

        # # Concatenate all of the output tensors from the EncoderRNN into a single tensor
        # encoder_outputs = torch.cat(encoder_outputs, dim=1)  # type: ignore

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
        embedded = torch.nn.functional.relu(embedded)

        # Perform a single forward pass through the RNN
        if self.type == 'LSTM':
            hidden_state, cell_state = hidden
            output, (hidden_state, cell_state) = self.cell(
                embedded, (hidden_state, cell_state))
            hidden_state = (hidden_state, cell_state)
        else:
            output, hidden_state = self.cell(embedded, hidden)
        return output, hidden_state


class Decoder(torch.nn.Module):
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
        super(Decoder, self).__init__()
        self.type = type_
        self.num_layers = num_layers_
        self.hidden_dim = hidden_dim_
        self.batch_first = batch_first_
        self.output_dim = output_dim_
        self.embed_dim = embed_dim_
        self.dropout_rate = 0 if num_layers_ <= 1 else dropout_rate_
        self.bidirectional = bidirectional_

        self.embedding = torch.nn.Embedding(  # type: torch.nn.Embedding
            self.output_dim, self.embed_dim)  # input_dim: int, embed_dim: int
        self.type = type_  # type: str
        self.cell = cell(type_)(  # type: torch.nn.RNNBase
            self.embed_dim, self.hidden_dim, num_layers=num_layers_, batch_first=batch_first_, dropout=self.dropout_rate, bidirectional=bidirectional_)
        self.out = torch.nn.Linear(  # type: torch.nn.Linear
            self.hidden_dim*(1+self.bidirectional), self.output_dim)  # in_features: int, out_features: int

    def greedy_or_gumbel_decode(self, encoder_outputs, encoder_hidden, target_tensor=None, teacher_ratio=0.5):
        """
        Perform decoding using either greedy or Gumbel-Softmax sampling.

        Args:
            encoder_outputs (torch.Tensor): The outputs from the encoder, with shape [batch_size, seq_len, hidden_size].
            encoder_hidden (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]): The hidden state from the encoder.
            target_tensor (torch.Tensor, optional): The target tensor for teacher forcing, with shape [batch_size, seq_len].
            teacher_ratio (float, optional): The probability of using teacher forcing, default is 0.5.

        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]: 
                - decoder_outputs: The outputs from the decoder, with shape [batch_size, seq_len, vocab_size].
                - decoder_hidden: The final hidden state of the decoder.
        """
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.full((batch_size, 1), dataset.start, dtype=torch.long, device=device)
        decoder_outputs = []

        if self.type == 'LSTM':
            encoder_hidden, encoder_cell = encoder_hidden
            decoder_hidden = self._match_hidden_shape(encoder_hidden)
            decoder_cell = self._match_hidden_shape(encoder_cell)
        else:
            decoder_hidden = self._match_hidden_shape(encoder_hidden)
            decoder_cell = None

        for i in range(config.MAX_LENGTH):
            if self.type == 'LSTM':
                decoder_output, (decoder_hidden, decoder_cell) = self.forward_step(decoder_input, (decoder_hidden, decoder_cell))
            else:
                decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            use_teacher = target_tensor is not None and teacher_ratio > random.random()
            if use_teacher:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                logits = decoder_output[:, -1]
                if self.use_gumbel:
                    gumbel_out = gumbel_softmax_sample(logits, temperature=self.temperature, hard=True)
                    token_ids = gumbel_out.argmax(dim=-1, keepdim=True)
                else:
                    token_ids = logits.argmax(dim=-1, keepdim=True)
                decoder_input = token_ids

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs, decoder_hidden
    
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
        if self.type == 'LSTM':
            encoder_hidden, encoder_cell = encoder_hidden
            if encoder_hidden.shape[0] != self.num_layers*(1+self.bidirectional):
                encoder_hidden = encoder_hidden.mean(0)
                encoder_hidden = torch.stack(
                    [encoder_hidden for i in range(self.num_layers*(1+self.bidirectional))])
                encoder_cell = encoder_cell.mean(0)
                encoder_cell = torch.stack(
                    [encoder_cell for i in range(self.num_layers*(1+self.bidirectional))])
            decoder_cell = encoder_cell
            decoder_hidden = encoder_hidden
        else:
            if encoder_hidden.shape[0] != self.num_layers*(1+self.bidirectional):
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
        input_: torch.Tensor,  # (B * beam, 1)
        hidden: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Performs a single forward step of the RNN cell, applying embedding and activation to the input, then passing it through the RNN (LSTM, GRU, or RNN) and output layer.

        Args:
            input_ (torch.Tensor): Input tensor of shape (B * beam, 1).
            hidden (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]): Hidden state (and cell state for LSTM).

        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]: Output tensor and updated hidden state(s).
        """
        embed = self.embedding(input_)
        active_embed = torch.nn.functional.relu(embed)

        if isinstance(self.cell, torch.nn.LSTM):
            hidden_state, cell_state = hidden
            hidden_state = hidden_state.contiguous()
            cell_state = cell_state.contiguous()
            output, (hidden_state, cell_state) = self.cell(
                active_embed, (hidden_state, cell_state)
            )
            output = self.out(output)
            return output, (hidden_state, cell_state)
        else:
            hidden = hidden.contiguous()
            output, hidden_state = self.cell(active_embed, hidden)
            output = self.out(output)
            return output, hidden_state


import heapq

def gumbel_softmax_sample(logits, temperature=1.0, hard=False):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize using straight-through.

    Args:
        logits: [batch_size, vocab_size]
        temperature: Gumbel-Softmax temperature
        hard: Whether to use straight-through estimation

    Returns:
        Sampled tensor: [batch_size, vocab_size]
    """
    gumbel_noise = -torch.empty_like(logits).exponential_().log()
    y = torch.nn.functional.softmax((logits + gumbel_noise) / temperature, dim=-1)

    if hard:
        # Straight-through: forward as one-hot, backward as soft
        index = y.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        y = (y_hard - y).detach() + y
    return y

class BeamSearchDecoder(Decoder):
    """
    An extension of the Decoder class that supports Beam Search decoding for inference.

    Beam search maintains multiple candidate sequences at each decoding time step and 
    chooses the most probable sequences based on cumulative scores.

    Args:
        beam_size (int): Number of beams to use for decoding. If 1, defaults to greedy decoding.
        use_gumbel (bool): If True, enables Gumbel-Softmax sampling for training (only used when target_tensor is provided).
        temperature (float): Temperature parameter for Gumbel-Softmax sampling.
        *args: Additional positional arguments passed to the base Decoder.
        **kwargs: Additional keyword arguments passed to the base Decoder.

    Attributes:
        beam_size (int): The number of beams maintained during decoding.
        use_gumbel (bool): Flag indicating whether to use Gumbel-Softmax sampling.
        temperature (float): Temperature parameter for Gumbel-Softmax sampling.

    Methods:
        forward(encoder_outputs, encoder_hidden, target_tensor=None, teacher_ratio=0.5):
            Performs decoding using greedy/Gumbel-Softmax or beam search depending on context.

        _match_hidden_shape(hidden):
            Adjusts hidden state shape to match the decoder's expected number of layers and directions.

        _expand_for_beam(tensor, beam_size):
            Expands hidden states or other tensors to accommodate beam_size during batched decoding.
    """
    def __init__(self, beam_size=3, use_gumbel=False, temperature=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beam_size = beam_size
        self.use_gumbel = use_gumbel
        self.temperature = temperature
        

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, teacher_ratio=0.5):
        """
        Perform decoding using either greedy decoding, Gumbel-Softmax sampling, or beam search.

        Args:
            encoder_outputs (torch.Tensor): Encoder outputs of shape (batch_size, seq_len, hidden_dim).
            encoder_hidden (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]): 
                Final hidden state from the encoder. For LSTM, this is a tuple (hidden, cell).
            target_tensor (Optional[torch.Tensor], optional): Ground truth target tensor used for teacher forcing.
                Shape: (batch_size, seq_len). If provided, greedy decoding or Gumbel-Softmax is used.
            teacher_ratio (float, optional): Probability of using teacher forcing during decoding. Defaults to 0.5.

        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
                - Output predictions of shape (batch_size, seq_len, output_dim) as one-hot vectors.
                - Final decoder hidden state(s).
        """
        if self.beam_size == 1 or target_tensor is not None:
            return self.greedy_or_gumbel_decode(encoder_outputs, encoder_hidden, target_tensor, teacher_ratio)

        # Batched beam search for inference
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.full((batch_size * self.beam_size, 1), dataset.start, dtype=torch.long, device=device)

        if self.type == 'LSTM':
            encoder_hidden, encoder_cell = encoder_hidden
            decoder_hidden = self._expand_for_beam(self._match_hidden_shape(encoder_hidden), self.beam_size)
            decoder_cell = self._expand_for_beam(self._match_hidden_shape(encoder_cell), self.beam_size)
        else:
            decoder_hidden = self._expand_for_beam(self._match_hidden_shape(encoder_hidden), self.beam_size)
            decoder_cell = None

        sequences = torch.full((batch_size * self.beam_size, 1), dataset.start, dtype=torch.long, device=device)
        scores = torch.zeros(batch_size * self.beam_size, device=device)
        is_finished = torch.zeros_like(scores, dtype=torch.bool)

        for _ in range(config.MAX_LENGTH):
            if self.type == 'LSTM':
                output, (decoder_hidden, decoder_cell) = self.forward_step(decoder_input, (decoder_hidden, decoder_cell))
            else:
                output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)

            log_probs = torch.nn.functional.log_softmax(output[:, -1], dim=-1)
            vocab_size = log_probs.size(1)
            next_scores, next_tokens = torch.topk(log_probs, self.beam_size, dim=-1)

            # Expand to compute all beam combinations
            scores = scores.view(batch_size, self.beam_size, 1)
            next_scores = next_scores.view(batch_size, self.beam_size, self.beam_size)
            total_scores = scores + next_scores

            flat_scores = total_scores.view(batch_size, -1)
            top_scores, top_indices = torch.topk(flat_scores, self.beam_size, dim=-1)

            beam_indices = top_indices // self.beam_size
            token_indices = top_indices % self.beam_size

            new_sequences = []
            new_decoder_input = []
            new_hidden = []
            new_cell = [] if decoder_cell is not None else None

            for b in range(batch_size):
                for i in range(self.beam_size):
                    old_idx = b * self.beam_size + beam_indices[b, i]
                    token = next_tokens[old_idx][token_indices[b, i]]
                    seq = torch.cat([sequences[old_idx], token.unsqueeze(0)], dim=0)
                    new_sequences.append(seq)
                    new_decoder_input.append(token.view(1))
                    new_hidden.append(decoder_hidden[:, old_idx:old_idx+1])
                    if decoder_cell is not None:
                        new_cell.append(decoder_cell[:, old_idx:old_idx+1])

            sequences = torch.stack(new_sequences)
            decoder_input = torch.stack(new_decoder_input).to(device)
            decoder_hidden = torch.cat(new_hidden, dim=1)
            if decoder_cell is not None:
                decoder_cell = torch.cat(new_cell, dim=1)

            scores = top_scores.view(-1)
            is_finished = is_finished | (sequences[:, -1] == dataset.end)
            if is_finished.all():
                break

        sequences = sequences.view(batch_size, self.beam_size, -1)
        final_scores = scores.view(batch_size, self.beam_size)
        best_indices = final_scores.argmax(dim=1)
        best_sequences = sequences[torch.arange(batch_size), best_indices]

        one_hot_outputs = torch.nn.functional.one_hot(best_sequences[:, 1:], num_classes=self.output_dim).float()
        return one_hot_outputs, decoder_hidden

    def _match_hidden_shape(self, hidden):
        """
        Adjusts the shape of the encoder's hidden state to match the expected number of layers and directions
        of the decoder.

        This is used when the encoder and decoder have mismatched configurations.

        Args:
            hidden (torch.Tensor): The hidden state tensor of shape (num_layers_encoder, batch_size, hidden_dim).

        Returns:
            torch.Tensor: Adjusted hidden state tensor of shape (self.num_layers * (1 + self.bidirectional), batch_size, hidden_dim).
        """
        if hidden.shape[0] != self.num_layers * (1 + self.bidirectional):
            mean = hidden.mean(0)
            return torch.stack([mean for _ in range(self.num_layers * (1 + self.bidirectional))])
        return hidden

    def _expand_for_beam(self, tensor, beam_size):
        """
        Expands a tensor to accommodate beam size for batched beam search decoding.

        Depending on the dimensionality of the tensor, this method repeats entries along the appropriate axis.

        Args:
            tensor (torch.Tensor): A tensor of shape [L, B, H] or [B, H] or [B].
            beam_size (int): Number of beams to expand to.

        Returns:
            torch.Tensor: Expanded tensor with batch dimension multiplied by beam_size.
        
        Raises:
            ValueError: If the tensor shape is not supported.
        """
        if tensor.dim() == 3:  # [L, B, H]
            L, B, H = tensor.shape
            tensor = tensor.unsqueeze(2).repeat(1, 1, beam_size, 1)
            return tensor.view(L, B * beam_size, H)
        elif tensor.dim() == 2:  # [B, H]
            B, H = tensor.shape
            tensor = tensor.unsqueeze(1).repeat(1, beam_size, 1)
            return tensor.view(B * beam_size, H)
        elif tensor.dim() == 1:  # [B]
            B = tensor.shape[0]
            tensor = tensor.unsqueeze(1).repeat(1, beam_size)
            return tensor.view(B * beam_size)
        else:
            raise ValueError("Unsupported tensor shape for beam expansion")