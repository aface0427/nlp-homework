"""
T5 Model for Neural Machine Translation
"""
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer

class T5NMT(nn.Module):
    def __init__(self, model_name_or_path="t5-small", device="cpu", freeze_layers=True):
        super().__init__()
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        
        if freeze_layers:
            # Freeze all parameters first
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Unfreeze the last two layers of the encoder and decoder
            # T5 structure: encoder.block, decoder.block
            # Each block has layers.
            
            # Unfreeze last 2 encoder blocks
            for i in range(len(self.model.encoder.block) - 2, len(self.model.encoder.block)):
                for param in self.model.encoder.block[i].parameters():
                    param.requires_grad = True
            
            # Unfreeze last 2 decoder blocks
            for i in range(len(self.model.decoder.block) - 2, len(self.model.decoder.block)):
                for param in self.model.decoder.block[i].parameters():
                    param.requires_grad = True
            
            # Also unfreeze the final layer norm and lm_head if desired?
            # Usually fine-tuning head is important.
            for param in self.model.lm_head.parameters():
                param.requires_grad = True
            for param in self.model.decoder.final_layer_norm.parameters():
                param.requires_grad = True
            for param in self.model.encoder.final_layer_norm.parameters():
                param.requires_grad = True
                
        self.model.to(device)
    
    def forward(self, src_text, tgt_text=None):
        """
        Args:
            src_text: List of source strings
            tgt_text: List of target strings (optional)
        """
        # Tokenize source
        # Requirement: "translate Chinese to English: " prefix
        # We assume src_text already has it or we add it here?
        # Let's add it here to be safe and consistent.
        prefix = "translate Chinese to English: "
        src_text_with_prefix = [prefix + text for text in src_text]
        
        inputs = self.tokenizer(
            src_text_with_prefix, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(self.device)
        
        if tgt_text is not None:
            # Tokenize target
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    tgt_text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=128
                ).to(self.device)
            
            # T5 handles shifting labels internally
            outputs = self.model(
                input_ids=inputs.input_ids, 
                attention_mask=inputs.attention_mask, 
                labels=labels.input_ids
            )
            return outputs.loss, outputs.logits
        else:
            # Inference
            generated_ids = self.model.generate(
                input_ids=inputs.input_ids, 
                attention_mask=inputs.attention_mask, 
                max_length=128
            )
            return generated_ids

def create_t5_model(model_name_or_path="t5-small", device="cpu"):
    return T5NMT(model_name_or_path, device)
