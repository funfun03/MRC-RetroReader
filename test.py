from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("funfun0803/retro_reader", use_auth_token=True)
tokenizer = AutoTokenizer.from_pretrained("funfun0803/retro_reader", use_auth_token=True)