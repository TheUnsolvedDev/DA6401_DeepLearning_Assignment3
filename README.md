# DA6401 Assignment 3

## This assignment encompasses several key objectives in the realm of sequence-to-sequence (seq2seq) learning and its advancements in natural language processing (NLP):

- **Modeling Sequence-to-Sequence Learning Problems using Recurrent Neural Networks (RNNs)**:
   This involves understanding how to set up a framework where the input and output are both sequences of data. RNNs are a natural choice for this task due to their ability to process sequences of inputs and outputs by maintaining a hidden state that captures information from previous inputs. Understanding how to structure the data, define the RNN architecture, and manage the training process is crucial.

- **Comparing Different Cells such as Vanilla RNN, LSTM, and GRU**:
   This objective entails examining and contrasting various types of RNN cells. Vanilla RNNs suffer from the vanishing gradient problem, limiting their ability to capture long-term dependencies. Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) cells were developed to address this issue by incorporating mechanisms to selectively retain or forget information over time. Comparing their performance in different tasks provides insight into their strengths and weaknesses.

- **Understanding How Attention Networks Overcome Limitations of Vanilla Seq2Seq Models**:
   Attention mechanisms enable the model to focus on different parts of the input sequence when generating each part of the output sequence, effectively aligning the input and output sequences. This helps overcome the limitations of vanilla seq2seq models, which often struggle with long input sequences or when the input and output sequences are of different lengths. Understanding how attention is integrated into seq2seq models and its impact on performance is crucial.

- **Understanding the Importance of Transformers in the Context of Machine Transliteration and NLP in General**:
   Transformers revolutionized NLP by introducing the self-attention mechanism, allowing the model to capture global dependencies in the input sequence more effectively. This architecture significantly outperformed traditional seq2seq models in tasks such as machine translation, text generation, and transliteration. Understanding the architecture of transformers, how they operate, and their advantages over traditional RNN-based models is essential in the context of machine transliteration and NLP tasks in general.

Overall, this assignment provides a comprehensive exploration of sequence-to-sequence learning, various RNN cell architectures, attention mechanisms, and the transformative impact of transformers in NLP tasks like machine transliteration.

# Vanilla model:

The initial part of the assignment involves implementing a vanilla sequence-to-sequence (seq2seq) model and tuning its hyperparameters using Bayesian optimization, facilitated by the WandB platform. Let's break down the process in detail:

1. **Vanilla Seq2Seq Model**:
   - Begin by setting up the basic architecture of the seq2seq model using a recurrent neural network (RNN), such as LSTM or GRU, as the core encoder-decoder framework.
   - Define the encoder and decoder components of the model. The encoder processes the input sequence and encodes it into a fixed-size context vector, while the decoder generates the output sequence based on this context vector.
   - Implement the training loop, which involves forward and backward passes through the network, computing gradients, and updating the model parameters using an optimization algorithm such as stochastic gradient descent (SGD) or Adam.

2. **Hyperparameter Tuning with Bayesian Sweep**:
   - Define the hyperparameters to be tuned, such as learning rate, batch size, dropout rate, etc. These hyperparameters significantly influence the performance of the model.
   - Set up a Bayesian sweep using the WandB platform. This involves specifying the hyperparameter search space and the optimization metric to be maximized (e.g., validation accuracy, BLEU score in machine translation tasks).
   - Run multiple training experiments with different hyperparameter configurations, recording the results (e.g., loss, validation accuracy) using WandB's tracking capabilities.
   - Utilize Bayesian optimization techniques to intelligently explore the hyperparameter space and identify promising configurations that yield optimal performance.
   - Continuously update the hyperparameter search based on the outcomes of previous experiments, gradually refining the search space to focus on the most promising regions.

3. **Analysis and Interpretation**:
   - Analyze the results of the hyperparameter tuning experiments to identify the hyperparameter configurations that yield the best performance on the validation set.
   - Visualize the performance metrics across different hyperparameter configurations to gain insights into the impact of individual hyperparameters on model performance.
   - Interpret the findings and draw conclusions regarding the optimal hyperparameter settings for the vanilla seq2seq model in the given task.
   - Discuss any observed trends or patterns in the hyperparameter search process and their implications for future model development and optimization.

By following this approach, we can systematically explore the hyperparameter space and identify the optimal configuration for the vanilla seq2seq model using Bayesian optimization with the assistance of the WandB platform. This process helps to maximize the model's performance and efficiency while gaining valuable insights into the interplay between hyperparameters and model performance.

```bash
# Vanilla model
cd Vanilla/
python3 train.py --sweep # to run the sweep
python3 train.py --help
# usage: train.py [-h] [-ep EPOCHS] [-lr LEARNING_RATE] [-bs BATCH_SIZE]
#                 [-pe PRINT_EVERY] [-hu HIDDEN_UNITS] [-ed EMBED_DIM]
#                 [-nel ENCODER_NUM_LAYERS] [-ndl DECODER_NUM_LAYERS]
#                 [-dp DROPOUT] [-cell CELL_TYPE] [-bid BIDIRECTIONAL]
#                 [-gpu GPU_NUMBER] [-wp WANDB_PROJECT] [-we WANDB_ENTITY] [-s]

# options:
#   -h, --help            show this help message and exit
#   -ep EPOCHS, --epochs EPOCHS
#                         number of epochs to train for
#   -lr LEARNING_RATE, --learning_rate LEARNING_RATE
#                         learning rate
#   -bs BATCH_SIZE, --batch_size BATCH_SIZE
#                         input batch size for training
#   -pe PRINT_EVERY, --print_every PRINT_EVERY
#                         number of batches to print the loss
#   -hu HIDDEN_UNITS, --hidden_units HIDDEN_UNITS
#                         number of hidden units
#   -ed EMBED_DIM, --embed_dim EMBED_DIM
#                         number of embedding dimensions
#   -nel ENCODER_NUM_LAYERS, --encoder_num_layers ENCODER_NUM_LAYERS
#                         number of encoder layers
#   -ndl DECODER_NUM_LAYERS, --decoder_num_layers DECODER_NUM_LAYERS
#                         number of decoder layers
#   -dp DROPOUT, --dropout DROPOUT
#                         dropout probability
#   -cell CELL_TYPE, --cell_type CELL_TYPE
#                         RNN cell type
#   -bid BIDIRECTIONAL, --bidirectional BIDIRECTIONAL
#                         whether to use bidirectional RNN
#   -gpu GPU_NUMBER, --gpu_number GPU_NUMBER
#                         gpu number
#   -wp WANDB_PROJECT, --wandb_project WANDB_PROJECT
#                         wandb project name
#   -we WANDB_ENTITY, --wandb_entity WANDB_ENTITY
#                         wandb entity name
#   -s, --sweep           Perform hyper parameter tuning
# this can be used for running the necessary output
```

# Attention model
The process for a sequence-to-sequence model with Bahdanau attention:

1. **Seq2Seq Model with Bahdanau Attention**:
   - Implement the core seq2seq architecture with Bahdanau attention mechanism. This involves modifying the vanilla seq2seq model to incorporate attention mechanisms, specifically Bahdanau attention.
   - Define the encoder and decoder components as before, but with the addition of attention mechanisms in both the encoder and decoder.
   - In the encoder, compute the context vector for each time step by considering the attention weights over the input sequence.
   - In the decoder, use the context vector and the previous hidden state to compute the attention scores over the encoder hidden states, generating a weighted context vector to inform the next prediction.

2. **Hyperparameter Tuning with Bayesian Sweep**:
   - Define the hyperparameters specific to the seq2seq model with Bahdanau attention. This may include attention mechanism-specific parameters such as attention hidden size, attention mechanism type, etc., in addition to the standard hyperparameters.
   - Set up a Bayesian sweep on WandB with the expanded hyperparameter search space, including both standard hyperparameters and attention mechanism-related hyperparameters.
   - Conduct multiple training experiments with different hyperparameter configurations, tracking the performance metrics using WandB.
   - Employ Bayesian optimization techniques to explore the hyperparameter space effectively, identifying configurations that maximize performance.
   - Continuously refine the search space based on the results of previous experiments, focusing on regions with promising hyperparameter configurations.

3. **Analysis and Interpretation**:
   - Analyze the results of the hyperparameter tuning experiments, paying particular attention to the impact of attention mechanism-related hyperparameters on model performance.
   - Visualize the performance metrics across different hyperparameter configurations, considering both standard hyperparameters and attention mechanism-related hyperparameters.
   - Interpret the findings to identify the optimal hyperparameter settings for the seq2seq model with Bahdanau attention in the given task.
   - Discuss any observed trends or patterns in the hyperparameter search process, emphasizing the role of attention mechanisms in enhancing model performance.
   - Reflect on the implications of the findings for future model development and optimization, highlighting the importance of attention mechanisms in sequence-to-sequence learning tasks.

By following this adapted approach, we can systematically explore the hyperparameter space and identify the optimal configuration for the seq2seq model with Bahdanau attention using Bayesian optimization with the assistance of the WandB platform. This process helps to maximize the model's performance while gaining insights into the impact of attention mechanisms on model behavior.

```bash
# Vanilla model
cd Attention/
python3 train.py --sweep # to run the sweep
python3 train.py --help
# usage: train.py [-h] [-ep EPOCHS] [-lr LEARNING_RATE] [-bs BATCH_SIZE]
#                 [-pe PRINT_EVERY] [-hu HIDDEN_UNITS] [-ed EMBED_DIM]
#                 [-nel ENCODER_NUM_LAYERS] [-ndl DECODER_NUM_LAYERS]
#                 [-dp DROPOUT] [-cell CELL_TYPE] [-bid BIDIRECTIONAL]
#                 [-gpu GPU_NUMBER] [-wp WANDB_PROJECT] [-we WANDB_ENTITY] [-s]

# options:
#   -h, --help            show this help message and exit
#   -ep EPOCHS, --epochs EPOCHS
#                         number of epochs to train for
#   -lr LEARNING_RATE, --learning_rate LEARNING_RATE
#                         learning rate
#   -bs BATCH_SIZE, --batch_size BATCH_SIZE
#                         input batch size for training
#   -pe PRINT_EVERY, --print_every PRINT_EVERY
#                         number of batches to print the loss
#   -hu HIDDEN_UNITS, --hidden_units HIDDEN_UNITS
#                         number of hidden units
#   -ed EMBED_DIM, --embed_dim EMBED_DIM
#                         number of embedding dimensions
#   -nel ENCODER_NUM_LAYERS, --encoder_num_layers ENCODER_NUM_LAYERS
#                         number of encoder layers
#   -ndl DECODER_NUM_LAYERS, --decoder_num_layers DECODER_NUM_LAYERS
#                         number of decoder layers
#   -dp DROPOUT, --dropout DROPOUT
#                         dropout probability
#   -cell CELL_TYPE, --cell_type CELL_TYPE
#                         RNN cell type
#   -bid BIDIRECTIONAL, --bidirectional BIDIRECTIONAL
#                         whether to use bidirectional RNN
#   -gpu GPU_NUMBER, --gpu_number GPU_NUMBER
#                         gpu number
#   -wp WANDB_PROJECT, --wandb_project WANDB_PROJECT
#                         wandb project name
#   -we WANDB_ENTITY, --wandb_entity WANDB_ENTITY
#                         wandb entity name
#   -s, --sweep           Perform hyper parameter tuning
# this can be used for running the necessary output
```

# The hyperparameter setup for both
parameters_dict:
  - epochs:
      values: [20, 40, 60]
  - batch_size:
      values: [64, 128, 256]
  - hidden_units:
      values: [128, 256, 512]
  - embed_dim:
      values: [32, 64, 128]
  - encoder_num_layers:
      values: [1, 2, 3]
  - decoder_num_layers:
      values: [1, 2, 3]
  - dropout:
      values: [0.1, 0.2, 0.3]
  - cell_type:
      values: ["RNN", "LSTM", "GRU"]
  - bidirectional:
      values: [true, false]
  - learning_rate:
      values: [0.0005, 0.001, 0.003]

Also Bayesian Method of sweep was used!
