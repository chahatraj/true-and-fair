import os
import torch
import torch.nn as nn
from transformers import AutoModel

NUM_LABELS = 2
BERT_ENCODER_OUTPUT_SIZE = 768  # 1024 #768
CLF_LAYER_1_DIM = 64
CLF_DROPOUT_PROB = 0.4
MODE = "fine-tune"  # pre-train or fine-tune
NAME = "distilbert-base-uncased"  # "roberta-large"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CKPT = "05-03-23-01_19_final_model_pretrain.pt"
# MODEL_PATH = os.path.join(os.path.dirname(os.getcwd()), "models", CKPT)

# class BertClassifier(nn.Module):
#     def __init__(self, name=NAME, mode=MODE, pretrained_checkpoint=None):
#         super(BertClassifier, self).__init__()
#         self.mode = mode
#         D_in, H, D_out = BERT_ENCODER_OUTPUT_SIZE, CLF_LAYER_1_DIM, NUM_LABELS

#         if pretrained_checkpoint is None:
#             self.bert = AutoModel.from_pretrained(NAME)
#             self.classifier = nn.Sequential(
#                 nn.Linear(D_in, H),
#                 nn.ReLU(),
#                 nn.Dropout(CLF_DROPOUT_PROB),
#                 nn.Linear(H, D_out),
#             )
#         else:
#             state_dict = torch.load(pretrained_checkpoint, map_location=device)
#             self.bert = AutoModel.from_pretrained(NAME, state_dict={k:v for k,v in state_dict.items() if 'bert' in k})
#             self.classifier = nn.Sequential(
#                 nn.Linear(state_dict['classifier.0.weight'].shape[1], state_dict['classifier.0.weight'].shape[0]),
#                 nn.ReLU(),
#                 nn.Dropout(CLF_DROPOUT_PROB),
#                 nn.Linear(state_dict['classifier.3.weight'].shape[1], state_dict['classifier.3.weight'].shape[0]),
#             )
#             self.classifier[0].weight.data = state_dict['classifier.0.weight']
#             self.classifier[0].bias.data = state_dict['classifier.0.bias']
#             self.classifier[3].weight.data = state_dict['classifier.3.weight']
#             self.classifier[3].bias.data = state_dict['classifier.3.bias']

#         if self.mode == "pre-train":
#             freeze_bert = True
#         else:
#             freeze_bert = False

#         # Freeze the BERT model
#         if freeze_bert:
#             for param in self.bert.parameters():
#                 param.requires_grad = False

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         last_hidden_state_cls = outputs[0][:, 0, :]
#         logits = self.classifier(last_hidden_state_cls)
#         return logits


class BertClassifier(nn.Module):
    def __init__(self, name=NAME, mode=MODE, pretrained_checkpoint=None):
        super(BertClassifier, self).__init__()
        self.mode = mode
        D_in, H, D_out = BERT_ENCODER_OUTPUT_SIZE, CLF_LAYER_1_DIM, NUM_LABELS
        if pretrained_checkpoint is None:
            self.bert = AutoModel.from_pretrained(NAME)
        else:
            state_dict = torch.load(pretrained_checkpoint, map_location=device)
            bert_state_dict = {k: v for k, v in state_dict.items() if "bert" in k}
            self.bert = AutoModel.from_pretrained(NAME, state_dict=bert_state_dict)
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(CLF_DROPOUT_PROB),
            nn.Linear(H, D_out),
        )

        if self.mode == "pre-train":
            freeze_bert = True
        else:
            freeze_bert = False

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits


if __name__ == "__main__":
    model = BertClassifier()
    model = model.to(device)
