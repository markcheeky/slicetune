# Slicetune [WIP]

A simple, flexible and non-invasive parameter-efficient method for finetuning large neural models.

With few lines of code, you can finetune only a desired fraction of parameters. The method is compatible with any with any `torch.nn.Optimizer` and with any `torch.nn.Module` that uses linear layers, such as (huggingface) transformers. After finetuning, you can "fuse" the trained parameters into the model and obtain exactly the same architecture as the original checkpoint you started from. You don't need to use this package during inference - you don't even have to have it installed during inference.


## Instalation
`pip install "slicetune[pretty] @ git+ssh://git@github.com/markcheeky/slicetune.git"`

TODO: publish on pypi

optional dependency `pretty` is for a pretty summary of finetuning parameters.


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


## Q&A

### Why not finetune the whole model?

Slicetuning belongs to a family of **parameter-efficient finetuning methods (PEFT)** that update only a fraction of parameters. There are multiple benefits:

1. **Finetuning a large model requires a lot of (gpu) memory.** PEFT dramatically decreases memory usage because the optimizer stores state only for the trained parameters.
1. **It prevents catastrophic forgetting.** Training of all parameters causes models to forget general knowledge from pretraining phase. This is similar to overfitting - it makes the model perform well on train data but badly on unseen data.
1. **It makes finetuning faster.** Fitting less parameters usually needs less iterations to converge
1. **It increases accuracy in low-data setting.** If your train set is small, you might actually get a more capable model by training only a part of the model. This is likely a consequence of item 2.
1. **It increases robustness on out-of-domain data.** This is also a consequence of item 2.


### How does slicetuning work?

The idea is to update only selected slices of model parameter tensors. However, torch optimizers can only update whole tensors, not slices. `slicetune` solves it by replacing torch layers with slicetune layers. They contain an extra smaller parameter `tuner` that is added to a slice of `weight` during `.forward()`. Now, you can optimize only the tuner parameters during training. After training, you can swap slicetune layers back with standard torch layers, and "fuse" `tuner` with `weight` and obtain a model with the exact same architecture it had before.


### Why slicetune layers instead of zeroing-out majority of `.grads` in optimizer before `optimizer.step()`?
Becase optimizing just the tuners inside slicetune layers requires less memory. Let's say we have 1024x1024 linear layer and want to update just 256x256 parameters (around 6%). In zeroing-out method, optimizer saves the state for a each 1024x1024 weight matrix. With slicetune layers, the optimizer only saves the state for the small 256x256 matrix.


### TODO
- [ ] include the "whole-columns" 
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
