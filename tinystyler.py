import torch
from transformers import T5ForConditionalGeneration, AutoModelForCausalLM

class TinyStyler(torch.nn.Module):
    def __init__(self, model_name, model_type='T5', use_style=False, ctrl_embed_dim=768):
        super().__init__()
        if model_type in ['T5', 'GPT']:
            self.model_type = model_type
        else:
            raise Exception("Unsupported model type: ", model_type)
        if model_type == 'T5':  
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.use_style = use_style
        if self.use_style:
            self.ctrl_embed_dim = ctrl_embed_dim
            if hasattr(self.model.config, 'd_model'):
                self.proj = torch.nn.Linear(
                    self.ctrl_embed_dim, self.model.config.d_model
                )
            elif hasattr(self.model.config, 'n_embd'):
                self.proj = torch.nn.Linear(
                    self.ctrl_embed_dim, self.model.config.n_embd
                )
            else:
                self.proj = torch.nn.Linear(
                    self.ctrl_embed_dim, self.model.config.hidden_size
                )

    def forward(self, input_ids, attention_mask, labels=None, style=None):
        if self.use_style:
            style_embed = self.proj(style).unsqueeze(1)

        input_embeds = self.model.get_input_embeddings()(input_ids)
        if self.use_style:
            input_embeds = torch.cat([style_embed, input_embeds], dim=1)
            attention_mask = torch.cat(
                [
                    torch.ones((input_embeds.shape[0], 1)).to(attention_mask.device),
                    attention_mask,
                ],
                dim=1,
            )
            if (self.model_type == 'GPT') and (labels is not None):
                # pad labels for extra token
                pad_label = torch.full((labels.size(0), 1), -100, device=labels.device)
                labels = torch.cat([pad_label, labels], dim=1)

        return self.model(
            inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels
        )

    def generate(self, input_ids, attention_mask, style=None, **kwargs):
        if self.use_style:
            style_embed = self.proj(style.unsqueeze(1))

        input_embeds = self.model.get_input_embeddings()(input_ids)
        if self.use_style:
            input_embeds = torch.cat([style_embed, input_embeds], dim=1)
            attention_mask = torch.cat(
                [
                    torch.ones((input_embeds.shape[0], 1)).to(attention_mask.device),
                    attention_mask,
                ],
                dim=1,
            )

        return self.model.generate(
            inputs_embeds=input_embeds, attention_mask=attention_mask, **kwargs
        )