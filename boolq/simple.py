from datasets import load_dataset
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping
from transformers import T5ForConditionalGeneration, AutoTokenizer
import pandas as pd
import torch
from boolq.t5_modules import MyLightningDataModule, MyLightningModel
import pytorch_lightning as pl

#%%
if __name__ == '__main__':
# %%

    dataset = load_dataset('super_glue', 'boolq')
    df_train, df_validation = [pd.concat({
        "source_text": dataset[sub].data.to_pandas().apply(
            lambda row: f"question: {row.question} passage: {row.passage}", axis=1
        ),
        "target_text": dataset[sub].data.to_pandas().label.map({0: "no", 1: "yes"}),
        "target_class": dataset[sub].data.to_pandas().label
    }, axis=1) for sub in "train validation".split()]


    #%%

    batch_size=4
    source_max_token_len=512
    target_max_token_len=2
    dataloader_num_workers=1
    early_stopping_patience_epochs=0
    logger = "default"
    max_epochs = 2
    precision=32

    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base').to(0)

    lightning_module = MyLightningModel(
        tokenizer=tokenizer,
        model=model,
        save_only_last_epoch=True
    )
    lightning_module.load_from_checkpoint("./checkpoints/lightning_logs/version_35/checkpoints/epoch=0-step=2356.ckpt", model=model, tokenizer=tokenizer)

    data_module = MyLightningDataModule(
                df_train,
                df_validation,
                tokenizer=tokenizer,
                batch_size=batch_size,
                source_max_token_len=source_max_token_len,
                target_max_token_len=target_max_token_len,
                num_workers=dataloader_num_workers
            )


    callbacks = [TQDMProgressBar(refresh_rate=5)]
    #
    if early_stopping_patience_epochs > 0:
        early_stop_callback = EarlyStopping(
                    monitor="val_loss",
                    min_delta=0.00,
                    patience=early_stopping_patience_epochs,
                    verbose=True,
                    mode="min",
                )
        callbacks.append(early_stop_callback)

    #         # add gpu support
    gpus = 1
    #
    #         # add logger
    loggers = True if logger == "default" else logger
    #
    #         # prepare trainer
    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        max_epochs=max_epochs,
        gpus=gpus,
        precision=precision,
        log_every_n_steps=1,
        default_root_dir="checkpoints",
        enable_checkpointing=True
    )
    #
    # # fit trainer
    # trainer.fit(lightning_module, data_module)
    trainer.validate(lightning_module, data_module)

#%%

# %%
# a = "question: does ethanol take more energy make that produces passage: All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned. All of these steps require resources and an infrastructure. The total amount of energy input into the process compared to the energy released by burning the resulting ethanol fuel is known as the energy balance (or ``energy returned on energy invested''). Figures compiled in a 2007 report by National Geographic Magazine point to modest results for corn ethanol produced in the US: one unit of fossil-fuel energy is required to create 1.3 energy units from the resulting ethanol. The energy balance for sugarcane ethanol produced in Brazil is more favorable, with one unit of fossil-fuel energy required to create 8 from the ethanol. Energy balance estimates are not easily produced, thus numerous such reports have been generated that are contradictory. For instance, a separate survey reports that production of ethanol from sugarcane, which requires a tropical climate to grow productively, returns from 8 to 9 units of energy for each unit expended, as compared to corn, which only returns about 1.34 units of fuel energy for each unit of energy expended. A 2006 University of California Berkeley study, after analyzing six separate studies, concluded that producing ethanol from corn uses much less petroleum than producing gasoline."

#%%
# import torch
# # model = T5ForConditionalGeneration.from_pretrained('t5-base').to(0)
# # tokenizer = AutoTokenizer.from_pretrained("t5-base")
# encoder_outputs = model.encoder(tokenizer(a, return_tensors="pt").input_ids, return_dict=True, output_hidden_states=True)
# decoder_input_ids = torch.tensor([[model._get_decoder_start_token_id()]])
# generated = model.greedy_search(decoder_input_ids, encoder_outputs=encoder_outputs, return_dict_in_generate=True, output_scores=True)
# tokenizer.decode(generated[0][0])
# %%
# from transformers import PreTrainedModel
# import torch
# from typing import Union, Tuple
# DecodedOutput = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
#
# @torch.no_grad()
# def greedy_decode(model: PreTrainedModel,
#                   input_ids: torch.Tensor,
#                   length: int,
#                   attention_mask: torch.Tensor = None,
#                   return_last_logits: bool = True) -> DecodedOutput:
#     decode_ids = torch.full((input_ids.size(0), 1),
#                             model.config.decoder_start_token_id,
#                             dtype=torch.long).to(input_ids.device)
#     encoder_outputs = model.get_encoder()(input_ids, attention_mask=attention_mask)
#     next_token_logits = None
#     for _ in range(length):
#         model_inputs = model.prepare_inputs_for_generation(
#             decode_ids,
#             encoder_outputs=encoder_outputs,
#             past=None,
#             attention_mask=attention_mask,
#             use_cache=True)
#         outputs = model(**model_inputs)  # (batch_size, cur_len, vocab_size)
#         next_token_logits = outputs[0][:, -1, :]  # (batch_size, vocab_size)
#         decode_ids = torch.cat([decode_ids,
#                                 next_token_logits.max(1)[1].unsqueeze(-1)],
#                                dim=-1)
#     if return_last_logits:
#         return decode_ids, next_token_logits
#     return decode_ids
