from datasets import load_dataset

dataset = load_dataset('super_glue', 'boolq')

# %%
import pandas as pd

df_train, df_validation, df_test = [pd.concat({
    "source_text": dataset['train'].data.to_pandas().apply(
        lambda row: f"question: {row.question} passage: {row.passage}", axis=1),
    "target_text": dataset['train'].data.to_pandas().label.map({0: "no", 1: "yes"}),
}, axis=1) for sub in "train validation test".split()]

# import
from boolq.simplet5 import SimpleT5

# instantiate
model = SimpleT5()

# load (supports t5, mt5, byT5 models)
model.from_pretrained("t5", "t5-base")

# train
model.train(train_df=df_train,  # pandas dataframe with 2 columns: source_text & target_text
            eval_df=df_validation,  # pandas dataframe with 2 columns: source_text & target_text
            source_max_token_len=512,
            target_max_token_len=1,
            batch_size=4,
            max_epochs=1,
            use_gpu=True,
            outputdir="outputs",
            early_stopping_patience_epochs=0,
            precision=32
            )

# load trained T5 model
# model.load_model("t5", "path/to/trained/model/directory", use_gpu=False)
#
# # predict
# model.predict("input text for prediction")
