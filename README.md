# Slicetune 🍕 [WIP]

A simple, flexible and non-invasive parameter-efficient method for finetuning large neural models.

With slicetune you can finetune only a desired fraction of parameters with a few lines of code. Slicetune is compatible with any torch optimizer and with any torch model that uses linear layers, such as (huggingface) transformers. After finetuning, you can "fuse" the trained parameters into the model and get exactly the same architecture as the original checkpoint you started from. You don't need to use this package during inference - you don't even have to have it installed during inference. 

<br/>


## Instalation

`pip install "slicetune[pretty] @ git+ssh://git@github.com/markcheeky/slicetune.git"`

TODO: publish on pypi

optional dependency `pretty` is for a pretty summary of finetuning parameters.

<br/>


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

<br/>

## Q&A

### Why not finetune the whole model?

Slicetuning belongs to a family of *parameter-efficient finetuning methods (PEFT)* that update only a fraction of parameters. There are multiple benefits:

1. **It dramatically decreases (gpu) memory usage.** A full finetuning of a large model requires a lot of memory for optimizer state. In PEFT, optimizer state can be multiple times smaller.
1. **It prevents catastrophic forgetting.** Training of all parameters makes models forget general knowledge from pretraining. This is similar to overfitting - it makes the model perform well on training data but badly on unseen data.
1. **It makes finetuning faster.** Fitting less parameters usually needs less iterations to converge.
1. **It increases accuracy in low-data setting.** If you have a small training set, you might get a more capable model by training only a part of the model. This is likely because of item 2.
1. **It increases robustness on out-of-domain data.** This is also likely because of item 2.


### How does slicetuning work?

The idea is to update only selected slices of model parameter tensors. However, torch optimizers can only update whole tensors, not slices. `slicetune` solves it by replacing torch layers with slicetune layers. They contain an extra smaller parameter `tuner` that is added to a slice of `weight` during `.forward()`. Now, you can optimize only the tuner parameters during training. After training, you can swap slicetune layers back with standard torch layers, and "fuse" `tuner` with `weight` and obtain a model with the exact same architecture it had before.


### How does slicetuning compare relative to other PEFT methods?

A quantitative comparison is WIP, but we can go through some practical differences.

<details>
<summary>
    
#### Slicetune vs Adapters
    
</summary>
    
[*adapter-transformers*](https://github.com/adapter-hub/adapter-transformers) is a popular alternative to slicetune. Adapters inject additional low-parameter layers to a model and finetune only those layers. Because they modify the model, a finetuned *Bert* using adapters != *Bert* from *transformers* library. For example, you cannot use standard `transformers` library to load the model using `AutoModel.from_pretrained('./your_adapter_checkpoint')`.

In order to deal with this, they maintain the library as a fork of *transformers* and try to keep it as compatible as possible while supporting as many features and models as possible. This has some disadvantages - you need to replace your installation of *transformers* with the fork and if it does not support a feature that you used, you are out of luck and need to implement it yourself. In addition, using other libraries or tools that depend on *transformers* can cause problems during 1) installation, because pip will try to download *transformers* and *adapter-transformers*, and 2) during runtime because *adapters-transformers* library overloads tha package name "transformers". This means that when other library runs `import transformers`, it will receive a different package than it expects. During inference, you will also need to use `adapter-transformers` library which can be inconvenient if you want to use *transformers* library in the codebase as well.

In contrast, *slicetune* allows you to fuse the changes and get back the exact same architecture as it was before. You don't need to make any changes to the inference codebase - when you load a model finetuned using *slicetune*, it only differs from the pretrained model in the weight values. It also means that if you can compile the pretrained model using `torch.jit.trace`, you can also compile the finetuned model.

</details>
<details>
<summary>
    
#### Slicetune vs LoRA
    
</summary>

[LoRA](https://github.com/microsoft/LoRA) is another popular alternative. Unlike *adapter-transformers*, it is not implemented as a fork of HF *transformers*, so avoids it avoids the issues mentioned above. LoRA uses a similar approach to slicetune because LoRA layers can be also "fused" standard torch layers, keeping a finetuned model compatible with original model class. A quantitative comparison has not been done yet. 

</details>
<details>
<summary>

#### Slicetune vs BitFit, Prefix-tuning and finetuning model head
    
</summary>

These PEFT methods are more similar to *slicetune*, because you they don't require any changes to architecture.
    
In BitFit, you only update the bias parameters, but you don't have a full control about how much you want to update. To give a concrete example, bias parameters are only 0.04% of all parameters in `xlm-roberta-base`. In constract, *slicetune* fives you larger control about how much of the model you want to update - for example `xlm-roberta-base` has 30% of all parameters in linear layers, so you can finetune anywhere from 0 to 30% of all model parameters with `slicetune` (I will consider adding slicetune.nn.Embedding layer in the future). Note that you can easily use BitFit inside *slicetune* library instead of or in addition to slicetuning. See the arguments of `slicetune.mark_for_training()` method.

Prefix-tuning finetunes a fixed-sized sequence of embeddings that you prepend to input embeddings. This influences what the model outputs, so you can treat the prefix as parameters and finetune them. Like BitFit, you don't have much control over how many parameters you want to train, because you can't make the prefix arbitrarily long. Also, you introduce a computational overhead (because time complexity of self-attention is quadratic w.r.t. sequence length) that will also be present during inference. In addition, you if your model supports input length up to 512 and you use 100 for prefix tuning, you decrease the maximum input size by 20%. You might also need to adjust your inference code for tasks like token classification, because some outputs correspond to artificial prefix instead of input tokens.

Finetuning model head is a simple finetuning method, when only the last part of the model is updated and the "backbone" is viewed as a frozen feature extractor. This method converges quickly but the changes to the model are "shallow". If the features extracted by the backbone are good for the end task, the method will perform well and if they are not, the head might not have enough expressive power to change it. However this method is convenient and does not require any changes to the inference code. I will consider adding some util functions to *slicetune* to support this method and make it more convenient.
    
But what about finetuning several full weight matrices of the model? Well, I would like to compare that to *slicetune* in the future, and if it performs well, I will add some util functions to the library to make it more convenient.

</details>

### Why slicetune layers instead of zeroing-out majority of `.grads` in optimizer before `optimizer.step()`?

Becase optimizing just the tuners inside slicetune layers requires less memory. Let's say we have 1024x1024 linear layer and want to update just 256x256 parameters (around 6%). In zeroing-out method, optimizer saves the state for a each 1024x1024 weight matrix. With slicetune layers, the optimizer only saves the state for the small 256x256 matrix.



<br/>

## TODO
- [ ] fix that pretty summary does not sum params to 100%
- [ ] include the "whole-columns" 
- [ ] benchmark the method
- [ ] write tests
- [ ] write docs
- [ ] write examples
- [ ] add loading and saving of only finetuned weights
- [ ] write a slicetune layer for Conv2d like LoRA does?
- [ ] write a slicetune layer for Embedding like LoRA does?


## DONE
- [x] slicetune.nn.Layer
- [x] slicetune.nn.Linear
- [x] util fn for patching model
- [x] util fn for marking parameters for training
- [x] util fn for fusing model
- [x] util fn for description
