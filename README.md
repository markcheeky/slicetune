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

You can also use slicetune layers directly when you build a model:

```python3
INPUT_DIM = 756
HIDDEN_DIM = 1024
OUTPUT_DIM = 20
TUNER_SIZE = 0.2

model = torch.nn.Sequential(
    slicetune.nn.Linear(INPUT_DIM, HIDDEN_DIM, tuner_size=TUNER_SIZE),
    torch.nn.ReLU(),
    slicetune.nn.Linear(HIDDEN_DIM, HIDDEN_DIM, tuner_size=TUNER_SIZE),
    torch.nn.ReLU(),
    slicetune.nn.Linear(HIDDEN_DIM, HIDDEN_DIM, tuner_size=TUNER_SIZE),
    torch.nn.ReLU(),
    slicetune.nn.Linear(HIDDEN_DIM, OUTPUT_DIM, tuner_size=TUNER_SIZE)
)

slicetune.mark_for_training(model)

```



## How does it work?

The idea is to update only selected slices of model parameter tensors. However, torch optimizers can only update a whole tensor, not slices. `slicetune` solves it by replacing `torch.nn.Linear` layers with `slicetune.nn.Linear` which contains an extra parameter tensor `tuner` which is added to a slice of the `weight` parameter during `.forward()`. Now, you can optimize only the tuner parameters during training. After training, you can replace back slicetune layers with standard layers, apply the `tuner` to the `weight` and obtain a model with exactly the same architecture it had before.


## Q&A

#### Why slicetune layers instead of zeroing-out majority of `.grads` in optimizer before `optimizer.step()`?
Becase optimizing just the tuners is requires less memory. Let's say we have 1024x1024 linear layer and want to update just 256x256 (around 6%) parameters. In zeroing-out method, optimizer saves the state for a each 1024x1024 weight matrix. With slicetune layers, the optimizer only saves the state for the small 256x256 matrix.


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
