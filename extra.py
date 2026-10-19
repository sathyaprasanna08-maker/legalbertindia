from transformers import Trainer
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_dir = "./ner_finetuned_model"   # or wherever your current model is
from transformers import Trainer
trainer = Trainer(model=None)
trainer.save_model("saved_legal_ner_model")
