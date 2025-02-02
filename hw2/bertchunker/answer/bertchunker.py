import os, sys, argparse, gzip, re, logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
import tqdm
import string
import random
import numpy as np
# from torchcrf import CRF

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# print(f"Using device: {device}")

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from collections import defaultdict
def get_keyboard_neighbors(ch):
    keyboard_mappings = None
    if keyboard_mappings is None or len(keyboard_mappings) != 26:
        keyboard_mappings = defaultdict(lambda: [])
        keyboard = ["qwertyuiop", "asdfghjkl*", "zxcvbnm***"]
        row = len(keyboard)
        col = len(keyboard[0])

        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]

        for i in range(row):
            for j in range(col):
                for k in range(4):
                    x_, y_ = i + dx[k], j + dy[k]
                    if (x_ >= 0 and x_ < row) and (y_ >= 0 and y_ < col):
                        if keyboard[x_][y_] == '*': continue
                        if keyboard[i][j] == '*': continue
                        keyboard_mappings[keyboard[i][j]].append(keyboard[x_][y_])

    if ch not in keyboard_mappings: return [ch]
    return keyboard_mappings[ch]

def read_conll(handle, input_idx=0, label_idx=2):
    conll_data = []
    contents = re.sub(r'\n\s*\n', r'\n\n', handle.read())
    contents = contents.rstrip()
    for sent_string in contents.split('\n\n'):
        annotations = list(zip(*[ word_string.split() for word_string in sent_string.split('\n') ]))
        assert(input_idx < len(annotations))
        if label_idx < 0:
            conll_data.append( annotations[input_idx] )
            logging.info("CoNLL: {}".format( " ".join(annotations[input_idx])))
        else:
            assert(label_idx < len(annotations))
            conll_data.append(( annotations[input_idx], annotations[label_idx] ))
            logging.info("CoNLL: {} ||| {}".format( " ".join(annotations[input_idx]), " ".join(annotations[label_idx])))
    return conll_data

def drop_a_char(sentence):
    words = sentence.split(" ")

    for _ in range(10):
        word_idx = random.randint(0, len(words)-1)
        if len(words[word_idx]) < 3: continue
        pos = random.randint(1, len(words[word_idx])-2)
        words[word_idx] = words[word_idx][:pos] + words[word_idx][pos+1:]
        sentence = " ".join(words)
        break
    return sentence

def swap_a_char(sentence):
    words = sentence.split(" ")
    for _ in range(100):
        word_idx = random.randint(0, len(words)-1)
        if len(words[word_idx]) <= 3: continue
        pos = random.randint(1, len(words[word_idx])-3)
        #words[word_idx] = words[word_idx][:pos] + words[word_idx][pos+1:]
        words[word_idx] = words[word_idx][:pos] + words[word_idx][pos:pos+2][::-1] + words[word_idx][pos+2:]
        sentence = " ".join(words)
        break
    return sentence

def key_a_char(sentence):
    words = sentence.split(" ")
    for _ in range(100):
        word_idx = random.randint(0, len(words)-1)
        if len(words[word_idx]) <= 3: continue
        pos = random.randint(1, len(words[word_idx])-2)
        neighboring_chars = get_keyboard_neighbors(words[word_idx][pos])
        random_neighbor = np.random.choice(neighboring_chars, 1)[0]
        words[word_idx] = words[word_idx][:pos] + random_neighbor + words[word_idx][pos+1:]
        sentence = " ".join(words)
        break
    return sentence

def add_a_char(sentence):
    words = sentence.split(" ")
    alphabets = "abcdefghijklmnopqrstuvwxyz"
    alphabets = [i for i in alphabets]
    for _ in range(100):
        word_idx = random.randint(0, len(words)-1)
        if len(words[word_idx]) <= 3: continue
        pos = random.randint(1, len(words[word_idx])-1)
        #words[word_idx] = words[word_idx][:pos] + words[word_idx][pos+1:]
        new_char = np.random.choice(alphabets, 1)[0]
        words[word_idx] = words[word_idx][:pos] + new_char + words[word_idx][pos:]
        sentence = " ".join(words)
        break
    return sentence

def introduce_noise(sentence, aug_ratio=0.1):
    noisy_sentence = []
    alphabet = string.ascii_lowercase

    if random.random() < aug_ratio:
        # noise_type = random.choice(['swap', 'drop', 'add', 'key']),aug_ratio=1.0
        # 5~> 94.6813
        # 6~> 94.6299
        # 10~> 94.3346

        noise_type = random.choice(['swap', 'drop', 'add', 'key'])
        for _ in range(5):
            j_sent = " ".join(sentence)
            if noise_type=='swap':
                j_sent = swap_a_char(j_sent)
            elif noise_type=='drop':
                j_sent = drop_a_char(j_sent)
            elif noise_type=='add':
                j_sent = add_a_char(j_sent)
            elif noise_type=='key':
                j_sent = key_a_char(j_sent)
            sentence = j_sent.split(" ")

        noisy_sentence = sentence
    else:
        j_sent = " ".join(sentence)
        noisy_sentence = j_sent.split(" ")

    return noisy_sentence


class TransformerModel(nn.Module):

    def __init__(
            self,
            basemodel,
            tagset_size,
            lr=5e-5
        ):
        
        # this wasn't in use, omit for now
        # torch.manual_seed(1)

        super(TransformerModel, self).__init__()
        self.basemodel = basemodel
        # the encoder will be a BERT-like model that receives an input text in subwords and maps each subword into
        # contextual representations
        self.encoder = None
        # the hidden dimension of the BERT-like model will be automatically set in the init function!
        self.encoder_hidden_dim = 0
        # The linear layer that maps the subword contextual representation space to tag space
        self.classification_head = None
        # The CRF layer on top of the classification head to make sure the model learns to move from/to relevant tags
        # self.crf_layer = None
        # optimizers will be initialized in the init_model_from_scratch function
        self.optimizers = None
        self.init_model_from_scratch(basemodel, tagset_size, lr)

    def init_model_from_scratch(self, basemodel, tagset_size, lr):
        # if we set output_hidden_states=True, we can retain hidden states for all layers
        self.encoder = AutoModel.from_pretrained(basemodel, output_hidden_states=True)
        self.encoder_hidden_dim = self.encoder.config.hidden_size

        # Mini-Transformer Configuration
        # self.num_heads = 8  # Number of attention heads for the mini-Transformer
        # self.dim_feedforward = 2048  # Feedforward dimension in the Transformer
        # self.num_layers = 2  # Number of Transformer layers
        # # Transformer encoder layer for the mini-Transformer
        # self.transformer_encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=self.encoder_hidden_dim,
        #     nhead=self.num_heads,
        #     dim_feedforward=self.dim_feedforward,
        #     dropout=0.1
        # )
        # self.transformer_encoder = nn.TransformerEncoder(
        #     self.transformer_encoder_layer,
        #     num_layers=self.num_layers
        # )
        # # Classification head (after the mini-Transformer)
        # self.classification_head = nn.Sequential( 
        #     nn.Dropout(p=0.1),
        #     nn.Linear(self.encoder_hidden_dim, tagset_size)  # Output to tagset size
        # )

        #self.classification_head = nn.Linear(self.encoder_hidden_dim, tagset_size)
        self.classification_head = nn.Sequential( 
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(self.encoder_hidden_dim * 4, 3072),
            # nn.Linear(self.encoder_hidden_dim, 3072),
            # nn.Dropout(p=0.1),
            nn.ReLU(),  
            nn.Linear(3072, 768),
            # nn.Dropout(p=0.1),
            nn.ReLU(), 
            nn.Linear(768, tagset_size) 
        )
        # TODO initialize self.crf_layer in here as well.
        # TODO modify the optimizers in a way that each model part is optimized with a proper learning rate!
        self.optimizers = [
            optim.Adam(
                list(self.encoder.parameters()),
                lr=lr
            ),
            optim.SGD(
                list(self.classification_head.parameters()),
                lr = 0.1
            )
        ]

    def forward(self, sentence_input):
        # encoded = self.encoder(sentence_input).last_hidden_state #[16,65,3072]
        # print(encoded.shape)

        # tag_space = self.classification_head(encoded)
        #print(tag_space.shape)

        """
            trick 2 - Use more than the last layer of the Transformer since lower layers 
            of a pre-trained LLM tend to reflect “syntax” while higher levels tend to reflect “semantics” 
            (waving hands profusely).
        """
        # Concatenate the last 4 hidden layers
        encoded = self.encoder(sentence_input).hidden_states
        concat_hidden_states = torch.cat(encoded[-4:], dim=-1)
        tag_space = self.classification_head(concat_hidden_states)

        # Average the last 4 hidden layers
        # encoded = self.encoder(sentence_input).hidden_states
        # avg_hidden_states = torch.mean(torch.stack(encoded[-4:]), dim=0)
        # tag_space = self.classification_head(avg_hidden_states)

        """ 
            trick 4.3 - Improve the classification head using mini-transformer 
        """
        # Extract hidden states from BERT encoder
        # encoded = self.encoder(sentence_input).last_hidden_state
        # # Transpose the input to [sequence_length, batch_size, hidden_dim] for Transformer input
        # encoded = encoded.permute(1, 0, 2)
        # # Pass through the mini-Transformer
        # transformer_output = self.transformer_encoder(encoded)
        # # Transpose back to [batch_size, sequence_length, hidden_dim]
        # transformer_output = transformer_output.permute(1, 0, 2)
        # # Classification head
        # tag_space = self.classification_head(transformer_output)

        tag_scores = F.log_softmax(tag_space, dim=-1)
        # TODO modify the tag_scores to use the parameters of the crf_layer
        return tag_scores

class FinetuneTagger:

    def __init__(
            self,
            modelfile,
            modelsuffix='.pt',
            basemodel='distilbert-base-uncased',
            trainfile=os.path.join('data', 'train.txt.gz'),
            epochs=5,
            batchsize=4,
            lr=5e-5
        ):
        # the input sentences will be handled using this object, you do not need to manually encode input sentence words
        self.tokenizer = AutoTokenizer.from_pretrained(basemodel)
        self.trainfile = trainfile
        self.modelfile = modelfile
        self.modelsuffix = modelsuffix
        self.basemodel = basemodel
        self.epochs = epochs
        self.batchsize = batchsize
        self.lr = lr
        self.training_data = []
        self.tag_to_ix = {}  # replace output labels / tags with an index
        self.ix_to_tag = []  # during inference we produce tag indices so we have to map it back to a tag
        self.model = None # setup the model in self.decode() or self.train()

    def load_training_data(self, trainfile):
        if trainfile[-3:] == '.gz':
            with gzip.open(trainfile, 'rt') as f:
                self.training_data = read_conll(f)
        else:
            with open(trainfile, 'r') as f:
                self.training_data = read_conll(f)

        for sent, tags in self.training_data:
            for tag in tags:
                if tag not in self.tag_to_ix:
                    self.tag_to_ix[tag] = len(self.tag_to_ix)
                    self.ix_to_tag.append(tag)

        logging.info("tag_to_ix:", self.tag_to_ix)
        logging.info("ix_to_tag:", self.ix_to_tag)

    def prepare_sequence(self, input_tokens_list, target_sequence=None):
        """
        The function that creates single example (input, target) training tensors or (input) inference tensors.
        """
        sentence_in = self.tokenizer(
            input_tokens_list,
            is_split_into_words=True,
            add_special_tokens=False,
            return_tensors="pt"
        )
        if target_sequence:
            subword_positions = sentence_in.encodings[0].word_ids
            idxs = [self.tag_to_ix[w] for w in target_sequence]
            target = [idxs[x] for x in subword_positions]
            return sentence_in, torch.tensor(target, dtype=torch.long)
        return sentence_in

    def argmax(self, model, seq):
        output = [[] for _ in seq]
        with torch.no_grad():
            inputs = self.prepare_sequence(seq).to(device)
            tag_scores = model(inputs.input_ids).squeeze(0)
            for i, word_id in enumerate(inputs.encodings[0].word_ids):
                output[word_id].append(self.ix_to_tag[int(tag_scores[i].argmax(dim=0))])
        # TODO do a better subword-to-word resolution than the following line
        # Take the first subword and use the output tag for that subword for entire word
        output = [item[0] for item in output]
        assert len(seq) == len(output)
        return output

    def train(self):
        self.load_training_data(self.trainfile)
        self.model = TransformerModel(self.basemodel, len(self.tag_to_ix), lr=self.lr).to(device)
        # TODO You may want to set the weights in the following line to increase the effect of
        #   gradients for infrequent labels and reduce the dominance of the frequent labels
        loss_function = nn.NLLLoss()
        self.model.train()
        loss = float("inf")
        total_loss = 0
        loss_count = 0
        for epoch in range(self.epochs):
            train_iterator = tqdm.tqdm(self.training_data)
            batch = []
            for tokenized_sentence, tags in train_iterator:
                # Step 1. Get our inputs ready for the network, that is, turn them into
                # Tensors of subword indices. Pre-trained transformer based models come with their fixed
                # input tokenizer which in our case will receive the words in a sentence and will convert the words list
                # into a list of subwords (e.g. you can look at https://aclanthology.org/P16-1162.pdf to get a better
                # understanding about BPE subword vocabulary creation technique).
                # The expected labels will be copied as many times as the size of the subwords list for each word and
                # returned in targets label.
                augmented_sentence = introduce_noise(tokenized_sentence, aug_ratio=1.0)  #Augmenting Data before batching
                batch.append(self.prepare_sequence(augmented_sentence, tags))
                if len(batch) < self.batchsize:
                    continue
                pad_id = self.tokenizer.pad_token_id
                o_id = self.tag_to_ix['O']
                max_len = max([x[1].size(0) for x in batch])
                # in the next two lines we pad the batch items so that each sequence comes to the same size before
                #  feeding the input batch to the model and calculating the loss over the target values.
                input_batch = [x[0].input_ids[0].tolist() + [pad_id] * (max_len - x[0].input_ids[0].size(0)) for x in batch]
                target_batch = [x[1].tolist() + [o_id] * (max_len - x[0].input_ids[0].size(0)) for x in batch]
                sentence_in = torch.LongTensor(input_batch).to(device)
                targets = torch.LongTensor(target_batch).to(device)
                # Step 2. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.model.zero_grad()
                # Step 3. Run our forward pass.
                tag_scores = self.model(sentence_in)
                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = loss_function(tag_scores.view(-1, len(self.tag_to_ix)), targets.view(-1))
                total_loss += loss.item()
                loss_count += 1
                loss.backward()
                # TODO you may want to freeze the BERT encoder for a couple of epochs
                #   and then start performing full fine-tuning.
                for optimizer in self.model.optimizers:
                    optimizer.step()
                # HINT: getting the value of loss below 2.0 might mean your model is moving in the right direction!
                train_iterator.set_description(f"loss: {total_loss/loss_count:.3f}")
                del batch[:]

            if epoch == self.epochs - 1:
                epoch_str = '' # last epoch so do not use epoch number in model filename
            else:
                epoch_str = str(epoch)
            savefile = self.modelfile + epoch_str + self.modelsuffix
            print(f"Saving model file: {savefile}", file=sys.stderr)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.model.optimizers[0].state_dict(),
                        'loss': loss,
                        'tag_to_ix': self.tag_to_ix,
                        'ix_to_tag': self.ix_to_tag,
                    }, savefile)

    def model_str(self):
        if not os.path.isfile(self.modelfile + self.modelsuffix):
            raise IOError(f"Error: missing model file {self.modelfile + self.modelsuffix}")

        saved_model = torch.load(self.modelfile + self.modelsuffix)
        tag_to_ix = saved_model['tag_to_ix']
        ix_to_tag = saved_model['ix_to_tag']
        model = TransformerModel(self.basemodel, len(tag_to_ix), lr=self.lr).to(device)
        model.load_state_dict(saved_model['model_state_dict'])
        return str(model)

    def decode(self, inputfile):
        if inputfile[-3:] == '.gz':
            with gzip.open(inputfile, 'rt') as f:
                input_data = read_conll(f, input_idx=0, label_idx=-1)
        else:
            with open(inputfile, 'r') as f:
                input_data = read_conll(f, input_idx=0, label_idx=-1)

        if not os.path.isfile(self.modelfile + self.modelsuffix):
            raise IOError(f"Error: missing model file {self.modelfile + self.modelsuffix}")

        saved_model = torch.load(self.modelfile + self.modelsuffix)
        self.tag_to_ix = saved_model['tag_to_ix']
        self.ix_to_tag = saved_model['ix_to_tag']
        model = TransformerModel(self.basemodel, len(self.tag_to_ix), lr=self.lr).to(device)
        model.load_state_dict(saved_model['model_state_dict'])
        # use the model for evaluation not training
        model.eval()
        decoder_output = []
        for sent in tqdm.tqdm(input_data):
            decoder_output.append(self.argmax(model, sent))
        return decoder_output

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--inputfile", dest="inputfile",
                            default=os.path.join('data', 'input', 'dev.txt'),
                             help="produce chunking output for this input file")
    argparser.add_argument("-t", "--trainfile", dest="trainfile",
                            default=os.path.join('data', 'train.txt.gz'),
                            help="training data for chunker")
    argparser.add_argument("-m", "--modelfile", dest="modelfile",
                            default=os.path.join('data', 'chunker'),
                            help="filename without suffix for model files")
    argparser.add_argument("-s", "--modelsuffix", dest="modelsuffix", default='.pt',
                            help="filename suffix for model files")
    argparser.add_argument("-M", "--basemodel", dest="basemodel",
                            default='distilbert-base-uncased',
                            help="The base huggingface pretrained model to be used as the encoder.")
    argparser.add_argument("-e", "--epochs", dest="epochs", type=int, default=5,
                            help="number of epochs [default: 5]")
    argparser.add_argument("-b", "--batchsize", dest="batchsize", type=int, default=16,
                            help="batch size [default: 16]")
    argparser.add_argument("-r", "--lr", dest="lr", type=float, default=5e-5,
                            help="the learning rate used to finetune the BERT-like encoder module.")
    argparser.add_argument("-f", "--force", dest="force", action="store_true", default=False,
                            help="force training phase (warning: can be slow)")
    argparser.add_argument("-l", "--logfile", dest="logfile", default=None,
                            help="log file for debugging")
    opts = argparser.parse_args()
    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)
    modelfile = opts.modelfile
    if modelfile.endswith('.pt'):
        modelfile = modelfile.removesuffix('.pt')
    chunker = FinetuneTagger(
                    modelfile,
                    modelsuffix=opts.modelsuffix,
                    basemodel=opts.basemodel,
                    trainfile=opts.trainfile,
                    epochs=opts.epochs,
                    batchsize=opts.batchsize,
                    lr=opts.lr
                )
    if not os.path.isfile(modelfile + opts.modelsuffix) or opts.force:
        print(f"Could not find modelfile {modelfile + opts.modelsuffix} or -f used. Starting training.", file=sys.stderr)
        chunker.train()
        print("Training done.", file=sys.stderr)
    # use the model file if available and opts.force is False
    assert(os.path.isfile(modelfile + opts.modelsuffix))
    print(f"Found modelfile {modelfile + opts.modelsuffix}. Starting decoding.", file=sys.stderr)
    decoder_output = chunker.decode(opts.inputfile)
    print("\n\n".join([ "\n".join(output) for output in decoder_output ]))
