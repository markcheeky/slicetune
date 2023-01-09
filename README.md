# Slicetune [WIP]

A simple, flexible and non-invasive parameter-efficient method for finetuning large neural models.


## Example usage

```python3
# create a model
model = transformers.AutoModel.from_pretrained("xlm-roberta-base")

# replace standard layers with equivalent slicetuning layers
slicetune.patch_linears(model, tuner_size=0.3)

# makes only slicetuners trainable
slicetune.mark_for_training(model)

# display info about trained params
print(slicetune.pretty_describe(model))

# finetune your model
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=5e-5)
...

# replace slicetuning layers with equivalent standard layers
slicetune.fuse(model)

# Your model is now 100% compatible with the original architecture.
# You can save it or load it using standard HF methods.
model.save_pretrained("./my_model")
model = transformers.AutoModel.from_pretrained("./my_model")
```


### TODO

- [ ] benchmark the method
- [ ] write tests
- [ ] write docs
- [ ] write examples
- [ ] add loading and saving of only finetuned weights
- [ ] write a slicetune layer for Conv2d like LoRA does?
- [ ] write a slicetune layer for Embedding like LoRA does?


### DONE
- [x] slicetune.nn.Layer
- [x] slicetune.nn.Linear
- [x] util fn for patching model
- [x] util fn for marking parameters for training
- [x] util fn for fusing model
- [x] util fn for description
