from faceAttr_trainer import Classifier_Trainer
import config as cfg

trainer = Classifier_Trainer(cfg.Epoches, cfg.Epoches, cfg.lr, cfg.model_type)
trainer.fit()
