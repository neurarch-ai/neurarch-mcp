# The static parser and linter against 59 real PyTorch repositories

**Date:** 2026-08-31
**Engine:** `src/vendor/engine.bundle.mjs`, first line `var __create = Object.create;`, 444458 bytes, sha256 `fba96901cdeced2da9ee60667a2c591f6f0d945a12011bb84c34d0f5b60426c3`. Provenance from `src/vendor/verifier.bundle.mjs` (`provenanceFor`).
**Command:**

```
node scripts/lint-real-repos.mjs --repos-dir <dir> --clone
```

`<dir>` receives one blob-less shallow clone per repo (`git clone --depth 1 --filter=blob:none --no-checkout`) with only the manifest files checked out. The manifest is `scripts/real-repos.json`, the hand verdicts are `scripts/real-repos-verdicts.json`, the machine output is `docs/real-repos-results.json`. The whole run (116 files) parses and lints in about 150 ms.

This is a measurement, not a demo. `neurarch-mcp` reads PyTorch source with a static, regex-and-tree parser and lints the graph it reconstructs. The question here is how much of real, popular, hand-written model code that parser can actually read today, and whether the findings it raises on such code are true. Failures are the result. Nothing below was filtered.

## Headline numbers

| Measure | Value |
|---|---|
| Repositories | 59 (all cloned; none skipped as dead) |
| Files attempted | 116 |
| Files that returned a graph | 73 (62.9%) |
| Files that returned nothing | 43 (40 `no-model`: a class was found but yielded zero layers; 3 `no-model-class`: no class defines both `__init__` and `forward`) |
| Repositories with at least one file parsed | 44 of 59 (74.6%) |
| Layers per parsed graph | min 1, **median 2**, max 141 |
| Parsed graphs with 3 layers or fewer | 48 of 73 |
| Parsed graphs with 10 layers or more | 9 of 73 (9 repos) |
| Nodes produced vs. `nn.*` constructor calls present in the parsed files | 450 of 794 |
| Parsed graphs carrying at least one unresolved parameter (source text such as `config.hidden_size` stored where a number should be) | 60 of 73 (228 params in total) |
| Findings | 1 block, 76 warn, 26 info |
| Parsed graphs with a block | 1 of 73 |
| Parsed graphs with no finding at all | 10 of 73 |

### Hand-judged precision

Every one of the 103 findings was opened against the source and judged as (a) a real issue, (b) a parser artefact, or (c) a true structural fact that is not a bug. Not a sample: all of them.

| Severity | Total | Real | Artefact | Not a bug |
|---|---|---|---|---|
| block | 1 | **0** | 1 | 0 |
| warn | 76 | **0** | 70 | 6 |
| info | 26 | **0** | 19 | 7 |
| all | 103 | **0** | 90 | 13 |

Precision on real code is zero. 87% of what the linter said was caused by the parser reading the code wrong, and the remaining 13% was true but intended.

The single biggest source is one rule: `invalid-output-shape` fired 54 times (71% of all warnings), every time because the graph carried an unresolved symbol (`config.vocab_size`, `dims[0]`, `self.inplanes`, `1 if aa_layer else 2`) that was then propagated against the parser's default input of `[1,28,28]`. That rule has a published crash-study measurement behind it, and in the 264-graph study it is a real crash predictor; on real code with symbolic dimensions it is pure noise, because the "invalid shape" is the parser's own unevaluated string. It should not fire on a dimension the parser did not resolve.

The two measured rules that fired at all were `invalid-output-shape` (54) and `deep-no-norm` (6). The one block (`linear-after-conv-no-flatten`, on the pytorch-tutorial CNN) was an artefact: the flatten is `out.reshape(out.size(0), -1)` in `forward`, and tensor-method reshapes are invisible to the parser.

### What "parsed" means here

A file counts as parsed when the engine returned a non-null graph with at least one layer. That is the engine's own definition and it is generous. The median graph has 2 layers. `LlamaForCausalLM` parsed to exactly one node (`lm_head`); `GPT` from nanoGPT parsed to two (`lm_head` twice, once from the weight-tying line). The engine picks the **last** class in the file that defines both `__init__` and `forward` as the model. On a Hugging Face `modeling_*.py` that is a task head at the bottom of the file, and its body (`self.model = LlamaModel(config)`) is a same-file class instance, which the parser does not inline. The 9 graphs of 10+ layers (DiT 141, MAE 35, CLIP 25, RWKV 14, x-transformers 14, Mamba 12, DETR transformer 11, MNIST 11, ESRGAN 11) are the only ones where "parsed" resembles what a person would mean by it.

So the honest single sentence is: **the parser produced something for 63% of files, produced a graph a person would recognise as the model for about 8% (9 of 116), and every finding it raised on the result was either its own artefact or a design choice.**

## Real bugs found

Zero.

No finding survived the hand check as a real defect in the code it was raised on. This section is empty by measurement, not by omission. The nearest candidates and why they fail:

- `linear-after-conv-no-flatten` (block) on `yunjey/pytorch-tutorial` `.../convolutional_neural_network/main.py`: the code flattens with `out.reshape(out.size(0), -1)` on line 54. Artefact.
- `pool-into-linear-no-flatten` on all three `kuangliu/pytorch-cifar` models: each has `out.view(out.size(0), -1)` between the pool and the linear. Artefact, three times.
- `redundant-activation` on `google/gemma_pytorch`: `tanh` then `softmax` with no linear between them is Gemma 2 final-logit soft-capping followed by sampling. Intended.
- `deep-no-norm` on `xinntao/ESRGAN`: ESRGAN removes BatchNorm on purpose, and the paper says so. Intended.

## Per-repo table

`Layers` is the node count of the returned graph. `nn.* calls` is the number of `nn.<Layer>(` constructor calls in the whole file (containers, `Parameter`, `functional` and `init` excluded), the rough ceiling a complete parse would reach. `Unresolved` is the number of parameters stored as unevaluated source text. `Verdicts` gives the hand judgement of the findings on that file, or, for unparsed files, the construct attributed as the cause.

| Repo | File | Status | Main class | Layers | nn.* calls | Unresolved | Findings | Verdicts |
|---|---|---|---|---|---|---|---|---|
| karpathy/nanoGPT | model.py | parsed | GPT | 2 | 12 | 4 | consecutive-linear-no-activation, invalid-output-shape | artefact=2 |
| karpathy/minGPT | mingpt/model.py | parsed | GPT | 1 | 14 | 2 | invalid-output-shape | artefact=1 |
| karpathy/nanochat | nanochat/gpt.py | parsed | GPT | 2 | 2 | 0 | attention-no-pe | artefact=1 |
| Lightning-AI/litgpt | litgpt/model.py | no-model | RMSNorm |  | 15 |  | main class RMSNorm (line 1301) yielded zero layers | the last class in the file is a norm or utility holding only nn.Parameter |
| huggingface/transformers | src/transformers/models/llama/modeling_llama.py | parsed | LlamaForCausalLM | 1 | 11 | 2 | invalid-output-shape | artefact=1 |
| huggingface/transformers | src/transformers/models/bert/modeling_bert.py | parsed | BertForQuestionAnswering | 2 | 36 | 2 | clean |  |
| huggingface/transformers | src/transformers/models/gpt2/modeling_gpt2.py | parsed | GPT2ForQuestionAnswering | 5 | 19 | 8 | dropout-before-bn, bn-at-output, invalid-output-shape | artefact=3 |
| huggingface/transformers | src/transformers/models/vit/modeling_vit.py | parsed | ViTForImageClassification | 1 | 16 | 2 | invalid-output-shape | artefact=1 |
| huggingface/transformers | src/transformers/models/whisper/modeling_whisper.py | parsed | WhisperForAudioClassification | 3 | 23 | 3 | bn-at-output, invalid-output-shape | artefact=2 |
| huggingface/transformers | src/transformers/models/qwen2/modeling_qwen2.py | parsed | Qwen2ForCausalLM | 1 | 11 | 2 | invalid-output-shape | artefact=1 |
| huggingface/transformers | src/transformers/models/mistral/modeling_mistral.py | parsed | MistralForCausalLM | 1 | 11 | 2 | invalid-output-shape | artefact=1 |
| huggingface/transformers | src/transformers/models/mixtral/modeling_mixtral.py | parsed | MixtralForCausalLM | 1 | 8 | 2 | invalid-output-shape | artefact=1 |
| huggingface/transformers | src/transformers/models/gemma/modeling_gemma.py | parsed | GemmaForCausalLM | 1 | 11 | 2 | invalid-output-shape | artefact=1 |
| huggingface/transformers | src/transformers/models/clip/modeling_clip.py | no-model | CLIPForImageClassification |  | 22 |  | main class CLIPForImageClassification (line 957) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| huggingface/pytorch-image-models | timm/models/vision_transformer.py | parsed | VisionTransformer | 3 | 14 | 4 | invalid-output-shape | artefact=1 |
| huggingface/pytorch-image-models | timm/models/resnet.py | parsed | ResNet | 5 | 14 | 6 | invalid-output-shape | artefact=1 |
| huggingface/pytorch-image-models | timm/models/efficientnet.py | no-model | EfficientNetFeatures |  | 1 |  | main class EfficientNetFeatures (line 356) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| huggingface/pytorch-image-models | timm/models/convnext.py | parsed | ConvNeXt | 2 | 3 | 6 | invalid-output-shape | artefact=1 |
| huggingface/pytorch-image-models | timm/models/swin_transformer.py | no-model | SwinTransformer |  | 6 |  | main class SwinTransformer (line 675) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| pytorch/vision | torchvision/models/resnet.py | parsed | ResNet | 5 | 9 | 3 | invalid-output-shape | artefact=1 |
| pytorch/vision | torchvision/models/vgg.py | parsed | VGG | 3 | 13 | 1 | clean |  |
| pytorch/vision | torchvision/models/densenet.py | parsed | DenseNet | 5 | 16 | 3 | invalid-output-shape | artefact=1 |
| pytorch/vision | torchvision/models/mobilenetv2.py | parsed | MobileNetV2 | 1 | 3 | 1 | dropout-at-output | artefact=1 |
| pytorch/vision | torchvision/models/vision_transformer.py | parsed | VisionTransformer | 1 | 10 | 1 | dropout-at-output | artefact=1 |
| facebookresearch/DiT | models.py | parsed | DiT | 141 | 14 | 0 | bn-after-activation, attention-no-pe, deep-attention-default-init | artefact=2 not-a-bug=1 |
| facebookresearch/dinov2 | dinov2/models/vision_transformer.py | no-model | DinoVisionTransformer |  | 0 |  | main class DinoVisionTransformer (line 45) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| facebookresearch/mae | models_mae.py | parsed | MaskedAutoencoderViT | 35 | 3 | 4 | deep-no-norm, invalid-output-shape | artefact=2 |
| facebookresearch/mae | models_vit.py | no-model-class |  |  | 0 |  | no class defines both __init__ and forward | no class defines both __init__ and forward (subclass overriding forward only, or pipeline module) |
| facebookresearch/detr | models/detr.py | parsed | MLP | 1 | 4 | 0 | clean |  |
| facebookresearch/detr | models/transformer.py | parsed | TransformerDecoderLayer | 11 | 21 | 15 | attention-no-pe, dropout-at-output, double-norm x2, invalid-output-shape | artefact=4 not-a-bug=1 |
| facebookresearch/detr | models/backbone.py | no-model | Joiner |  | 0 |  | main class Joiner (line 96) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| facebookresearch/segment-anything | segment_anything/modeling/image_encoder.py | no-model | PatchEmbed |  | 5 |  | main class PatchEmbed (line 364) yielded zero layers | constructor call split across lines (black-formatted nn.Conv2d(\n ...)) |
| facebookresearch/segment-anything | segment_anything/modeling/sam.py | no-model | Sam |  | 0 |  | main class Sam (line 18) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| facebookresearch/segment-anything | segment_anything/modeling/mask_decoder.py | no-model | MLP |  | 5 |  | main class MLP (line 154) yielded zero layers | container built imperatively (ModuleList().append in a loop, nn.Sequential(*list), generator over zip) |
| facebookresearch/ConvNeXt | models/convnext.py | no-model | LayerNorm |  | 8 |  | main class LayerNorm (line 119) yielded zero layers | the last class in the file is a norm or utility holding only nn.Parameter |
| facebookresearch/deit | models.py | parsed | DistilledVisionTransformer | 2 | 1 | 4 | consecutive-linear-no-activation, invalid-output-shape | artefact=2 |
| lucidrains/vit-pytorch | vit_pytorch/vit.py | parsed | ViT | 4 | 18 | 3 | invalid-output-shape | artefact=1 |
| lucidrains/vit-pytorch | vit_pytorch/simple_vit.py | parsed | SimpleViT | 2 | 13 | 2 | invalid-output-shape | artefact=1 |
| lucidrains/x-transformers | x_transformers/x_transformers.py | parsed | XTransformer | 14 | 49 | 6 | deep-no-norm, invalid-output-shape | artefact=2 |
| lucidrains/denoising-diffusion-pytorch | denoising_diffusion_pytorch/denoising_diffusion_pytorch.py | no-model | GaussianDiffusion |  | 20 |  | main class GaussianDiffusion (line 478) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| lucidrains/PaLM-pytorch | palm_pytorch/palm_pytorch.py | parsed | ParallelTransformerBlock | 4 | 5 | 4 | invalid-output-shape x2 | artefact=2 |
| openai/whisper | whisper/model.py | parsed | Whisper | 3 | 2 | 2 | redundant-activation, invalid-output-shape | artefact=2 |
| openai/CLIP | clip/model.py | parsed | CLIP | 25 | 32 | 13 | consecutive-linear-no-activation x3, invalid-output-shape x2 | artefact=5 |
| openai/guided-diffusion | guided_diffusion/unet.py | parsed | EncoderUNetModel | 1 | 18 | 1 | clean |  |
| state-spaces/mamba | mamba_ssm/modules/mamba_simple.py | parsed | Mamba | 12 | 6 | 2 | deep-no-norm, invalid-output-shape | artefact=1 not-a-bug=1 |
| state-spaces/mamba | mamba_ssm/models/mixer_seq_simple.py | parsed | MambaLMHeadModel | 1 | 2 | 2 | invalid-output-shape | artefact=1 |
| BlinkDL/RWKV-LM | RWKV-v4neo/src/model.py | parsed | RWKV | 14 | 46 | 14 | consecutive-linear-no-activation x3, invalid-output-shape | artefact=4 |
| BlinkDL/RWKV-LM | RWKV-v7/rwkv_v7_demo.py | parsed | RWKV | 9 | 15 | 5 | invalid-output-shape | artefact=1 |
| deepseek-ai/DeepSeek-V3 | inference/model.py | parsed | Transformer | 2 | 0 | 0 | bn-at-output | artefact=1 |
| mistralai/mistral-inference | src/mistral_inference/transformer.py | parsed | Transformer | 1 | 2 | 2 | invalid-output-shape | artefact=1 |
| meta-llama/llama | llama/model.py | no-model | Transformer |  | 0 |  | main class Transformer (line 413) yielded zero layers | container built imperatively (ModuleList().append in a loop, nn.Sequential(*list), generator over zip) |
| meta-llama/llama3 | llama/model.py | no-model | Transformer |  | 0 |  | main class Transformer (line 251) yielded zero layers | container built imperatively (ModuleList().append in a loop, nn.Sequential(*list), generator over zip) |
| EleutherAI/gpt-neox | megatron/model/gpt2_model.py | no-model-class |  |  | 0 |  | no class defines both __init__ and forward | no class defines both __init__ and forward (subclass overriding forward only, or pipeline module) |
| EleutherAI/gpt-neox | megatron/model/transformer.py | no-model | NormPipe |  | 1 |  | main class NormPipe (line 1204) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| jzhang38/TinyLlama | lit_gpt/model.py | parsed | LLaMAMLP | 1 | 9 | 0 | clean |  |
| CompVis/stable-diffusion | ldm/modules/diffusionmodules/openaimodel.py | parsed | EncoderUNetModel | 1 | 20 | 1 | clean |  |
| CompVis/stable-diffusion | ldm/modules/attention.py | no-model | SpatialTransformer |  | 22 |  | main class SpatialTransformer (line 218) yielded zero layers | constructor call split across lines (black-formatted nn.Conv2d(\n ...)) |
| lllyasviel/ControlNet | cldm/cldm.py | no-model | ControlNet |  | 8 |  | main class ControlNet (line 48) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| ultralytics/ultralytics | ultralytics/nn/tasks.py | no-model | Ensemble |  | 7 |  | main class Ensemble (line 1502) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| ultralytics/ultralytics | ultralytics/nn/modules/block.py | no-model | Proto26 |  | 53 |  | main class Proto26 (line 1988) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| WongKinYiu/yolov7 | models/yolo.py | no-model | Model |  | 8 |  | main class Model (line 508) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| WongKinYiu/yolov7 | models/common.py | no-model | ST2CSPC |  | 59 |  | main class ST2CSPC (line 2001) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| milesial/Pytorch-UNet | unet/unet_model.py | no-model | UNet |  | 0 |  | main class UNet (line 6) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| milesial/Pytorch-UNet | unet/unet_parts.py | parsed | OutConv | 1 | 10 | 2 | invalid-output-shape | artefact=1 |
| kuangliu/pytorch-cifar | models/resnet.py | parsed | ResNet | 4 | 17 | 2 | pool-into-linear-no-flatten, invalid-output-shape | artefact=2 |
| kuangliu/pytorch-cifar | models/vgg.py | parsed | VGG | 6 | 6 | 2 | pool-into-linear-no-flatten, invalid-output-shape | artefact=2 |
| kuangliu/pytorch-cifar | models/densenet.py | parsed | DenseNet | 5 | 9 | 3 | invalid-output-shape | artefact=1 |
| kuangliu/pytorch-cifar | models/mobilenetv2.py | parsed | MobileNetV2 | 9 | 13 | 1 | pool-into-linear-no-flatten, invalid-output-shape | artefact=2 |
| pytorch/examples | mnist/main.py | parsed | Net | 11 | 6 | 0 | deep-no-norm | not-a-bug=1 |
| pytorch/examples | word_language_model/model.py | no-model | TransformerModel |  | 7 |  | main class TransformerModel (line 107) yielded zero layers | a triple-quoted string spanning lines inside a call in __init__ (a raise ValueError with a triple-quoted message) aborts the parse of the whole file |
| jadore801120/attention-is-all-you-need-pytorch | transformer/Models.py | parsed | Transformer | 1 | 7 | 2 | invalid-output-shape | artefact=1 |
| jadore801120/attention-is-all-you-need-pytorch | transformer/SubLayers.py | parsed | PositionwiseFeedForward | 5 | 10 | 6 | dropout-before-bn, bn-at-output, consecutive-linear-no-activation, invalid-output-shape | artefact=2 not-a-bug=2 |
| jadore801120/attention-is-all-you-need-pytorch | transformer/Layers.py | parsed | DecoderLayer | 2 | 0 | 0 | attention-no-pe | not-a-bug=1 |
| hyunwoongko/transformer | models/model/transformer.py | no-model | Transformer |  | 0 |  | main class Transformer (line 13) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| hyunwoongko/transformer | models/model/encoder.py | parsed | Encoder | 6 | 0 | 0 | clean |  |
| hyunwoongko/transformer | models/blocks/encoder_layer.py | parsed | EncoderLayer | 3 | 2 | 2 | attention-no-pe, dropout-at-output | artefact=1 not-a-bug=1 |
| hyunwoongko/transformer | models/layers/multi_head_attention.py | parsed | MultiHeadAttention | 4 | 4 | 8 | consecutive-linear-no-activation x3, invalid-output-shape | artefact=4 |
| kyegomez/BitNet | bitnet/bit_transformer.py | parsed | BitNetTransformer | 3 | 4 | 3 | invalid-output-shape | artefact=1 |
| kyegomez/BitNet | bitnet/bitlinear.py | no-model-class |  |  | 0 |  | no class defines both __init__ and forward | no class defines both __init__ and forward (subclass overriding forward only, or pipeline module) |
| Dao-AILab/flash-attention | flash_attn/models/gpt.py | parsed | GPTLMHeadModel | 3 | 3 | 4 | consecutive-linear-no-activation, invalid-output-shape | artefact=1 not-a-bug=1 |
| Dao-AILab/flash-attention | flash_attn/modules/mha.py | parsed | ParallelMHA | 8 | 11 | 0 | deep-no-norm, duplicate-positional-encoding | artefact=1 not-a-bug=1 |
| NVIDIA/Megatron-LM | megatron/core/models/gpt/gpt_model.py | parsed | GPTModel | 1 | 0 | 0 | clean |  |
| NVIDIA/Megatron-LM | megatron/core/transformer/transformer_layer.py | no-model | TransformerLayer |  | 0 |  | main class TransformerLayer (line 318) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| NVIDIA/Megatron-LM | megatron/core/transformer/attention.py | no-model | Attention |  | 0 |  | main class Attention (line 289) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| microsoft/LoRA | loralib/layers.py | no-model | ConvLoRA |  | 1 |  | main class ConvLoRA (line 246) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| ashawkey/stable-dreamfusion | nerf/network.py | parsed | NeRFNetwork | 1 | 7 | 0 | clean |  |
| ashawkey/stable-dreamfusion | nerf/renderer.py | no-model | NeRFRenderer |  | 0 |  | main class NeRFRenderer (line 257) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| huggingface/diffusers | src/diffusers/models/unets/unet_2d.py | parsed | UNet2DModel | 7 | 6 | 13 | invalid-output-shape | artefact=1 |
| huggingface/diffusers | src/diffusers/models/attention.py | no-model | FeedForward |  | 26 |  | main class FeedForward (line 1682) yielded zero layers | container built imperatively (ModuleList().append in a loop, nn.Sequential(*list), generator over zip) |
| huggingface/diffusers | src/diffusers/models/unets/unet_2d_condition.py | no-model | UNet2DConditionModel |  | 6 |  | main class UNet2DConditionModel (line 76) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| open-mmlab/mmdetection | mmdet/models/backbones/resnet.py | no-model | ResNet |  | 7 |  | main class ResNet (line 306) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| open-mmlab/mmdetection | mmdet/models/backbones/swin.py | parsed | SwinTransformer | 2 | 6 | 1 | dropout-at-output | artefact=1 |
| pytorch/audio | src/torchaudio/models/wav2vec2/model.py | no-model | HuBERTPretrainModel |  | 2 |  | main class HuBERTPretrainModel (line 123) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| pytorch/audio | src/torchaudio/models/wav2vec2/components.py | no-model | LogitGenerator |  | 20 |  | main class LogitGenerator (line 1104) yielded zero layers | torch.nn.X fully-qualified prefix instead of nn.X |
| pytorch/audio | src/torchaudio/models/conformer.py | no-model | Conformer |  | 19 |  | main class Conformer (line 215) yielded zero layers | torch.nn.X fully-qualified prefix instead of nn.X |
| speechbrain/speechbrain | speechbrain/lobes/models/transformer/Transformer.py | no-model | NormalizedEmbedding |  | 6 |  | main class NormalizedEmbedding (line 966) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| speechbrain/speechbrain | speechbrain/nnet/attention.py | parsed | RoPEMHA | 2 | 24 | 2 | attention-no-pe, invalid-output-shape | artefact=2 |
| rwightman/gen-efficientnet-pytorch | geffnet/gen_efficientnet.py | parsed | GenEfficientNet | 3 | 4 | 2 | invalid-output-shape | artefact=1 |
| rwightman/gen-efficientnet-pytorch | geffnet/mobilenetv3.py | parsed | MobileNetV3 | 2 | 5 | 2 | invalid-output-shape | artefact=1 |
| xinntao/ESRGAN | RRDBNet_arch.py | parsed | RRDBNet | 11 | 13 | 12 | deep-no-norm, invalid-output-shape | artefact=1 not-a-bug=1 |
| NVlabs/stylegan2-ada-pytorch | training/networks.py | parsed | Discriminator | 2 | 0 | 0 | clean |  |
| facebookresearch/fairseq | fairseq/models/transformer/transformer_encoder.py | no-model | TransformerEncoderBase |  | 1 |  | main class TransformerEncoderBase (line 37) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| facebookresearch/fairseq | fairseq/modules/multihead_attention.py | no-model | MultiheadAttention |  | 4 |  | main class MultiheadAttention (line 63) yielded zero layers | constructor call split across lines (black-formatted nn.Conv2d(\n ...)) |
| facebookresearch/fairseq | fairseq/models/roberta/model.py | parsed | RobertaEncoder | 1 | 6 | 2 | invalid-output-shape | artefact=1 |
| ultralytics/yolov5 | models/yolo.py | no-model | DetectionModel |  | 3 |  | main class DetectionModel (line 215) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| ultralytics/yolov5 | models/common.py | parsed | Classify | 3 | 20 | 3 | invalid-output-shape | artefact=1 |
| microsoft/Swin-Transformer | models/swin_transformer.py | parsed | SwinTransformer | 1 | 13 | 2 | invalid-output-shape | artefact=1 |
| google/gemma_pytorch | gemma/model.py | parsed | GemmaForCausalLM | 2 | 0 | 0 | output-activation, redundant-activation | not-a-bug=2 |
| black-forest-labs/flux | src/flux/model.py | parsed | Flux | 2 | 2 | 4 | consecutive-linear-no-activation, invalid-output-shape | artefact=2 |
| black-forest-labs/flux | src/flux/modules/layers.py | parsed | LastLayer | 2 | 24 | 3 | invalid-output-shape | artefact=1 |
| mosaicml/llm-foundry | llmfoundry/models/mpt/modeling_mpt.py | no-model | ComposerMPTCausalLM |  | 4 |  | main class ComposerMPTCausalLM (line 1454) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| eriklindernoren/PyTorch-GAN | implementations/dcgan/dcgan.py | no-model | Discriminator |  | 19 |  | main class Discriminator (line 73) yielded zero layers | container built imperatively (ModuleList().append in a loop, nn.Sequential(*list), generator over zip) |
| eriklindernoren/PyTorch-GAN | implementations/cyclegan/models.py | no-model | Discriminator |  | 26 |  | main class Discriminator (line 95) yielded zero layers | container built imperatively (ModuleList().append in a loop, nn.Sequential(*list), generator over zip) |
| yunjey/pytorch-tutorial | tutorials/02-intermediate/convolutional_neural_network/main.py | parsed | ConvNet | 3 | 10 | 2 | linear-after-conv-no-flatten (block), invalid-output-shape | artefact=2 |
| yunjey/pytorch-tutorial | tutorials/03-advanced/image_captioning/model.py | parsed | DecoderRNN | 3 | 5 | 7 | invalid-output-shape | artefact=1 |
| microsoft/unilm | beit/modeling_finetune.py | parsed | VisionTransformer | 1 | 11 | 2 | invalid-output-shape | artefact=1 |

## Findings by rule, with verdicts

`Measured` means `provenanceFor` returns a published measurement for the rule.

| Rule | Severity | Measured | Count | Real | Artefact | Not a bug | Unjudged |
|---|---|---|---|---|---|---|---|
| invalid-output-shape | warn | yes | 54 | 0 | 54 | 0 | 0 |
| consecutive-linear-no-activation | info | no | 14 | 0 | 13 | 1 | 0 |
| attention-no-pe | warn | no | 6 | 0 | 3 | 3 | 0 |
| deep-no-norm | info | yes | 6 | 0 | 2 | 4 | 0 |
| dropout-at-output | warn | no | 5 | 0 | 5 | 0 | 0 |
| bn-at-output | warn | no | 4 | 0 | 3 | 1 | 0 |
| pool-into-linear-no-flatten | warn | no | 3 | 0 | 3 | 0 | 0 |
| dropout-before-bn | info | no | 2 | 0 | 1 | 1 | 0 |
| double-norm | info | no | 2 | 0 | 2 | 0 | 0 |
| redundant-activation | warn | no | 2 | 0 | 1 | 1 | 0 |
| bn-after-activation | warn | no | 1 | 0 | 1 | 0 | 0 |
| deep-attention-default-init | info | no | 1 | 0 | 0 | 1 | 0 |
| duplicate-positional-encoding | info | no | 1 | 0 | 1 | 0 | 0 |
| output-activation | info | no | 1 | 0 | 0 | 1 | 0 |
| linear-after-conv-no-flatten | block | no | 1 | 0 | 1 | 0 | 0 |

The full per-finding verdict, with a one-line reason each, is in `scripts/real-repos-verdicts.json` (keyed `repo|file|rule|component`) and merged into every record of `docs/real-repos-results.json`.

The 13 "not a bug" verdicts are all of one shape: the parser read a sublayer, a mixer, an attention module or a sampler as if it were a whole model, and the linter then correctly observed that a sublayer has no positional encoding, that an FFN sublayer ends in a norm, or that a sampler ends in a softmax. Those are true statements about the wrong scope.

## What the parser cannot read yet

Each of the 43 unparsed files was attributed to one primary construct by reading the `__init__` of the class the engine selected. Each construct was then reproduced in isolation against the engine so that the attribution is verified, not inferred. The minimal cases live in `scripts/real-repos-probes.mjs` (`node scripts/real-repos-probes.mjs` prints what the engine returns next to what a correct parse would return, 35 cases), so every item below can be fixed against a ten-line input.

| Unparsed file cause | Files |
|---|---|
| sub-modules are instances of classes from other files, or are passed in through __init__ | 13 |
| sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) | 13 |
| container built imperatively (ModuleList().append in a loop, nn.Sequential(*list), generator over zip) | 6 |
| no class defines both __init__ and forward (subclass overriding forward only, or pipeline module) | 3 |
| constructor call split across lines (black-formatted nn.Conv2d(\n ...)) | 3 |
| the last class in the file is a norm or utility holding only nn.Parameter | 2 |
| torch.nn.X fully-qualified prefix instead of nn.X | 2 |
| a triple-quoted string spanning lines inside a call in __init__ (a raise ValueError with a triple-quoted message) aborts the parse of the whole file | 1 |

Verified against the engine with minimal inputs, in order of how much real code they block:

1. **A same-file helper class instantiated directly is dropped, not inlined.** `self.inc = DoubleConv(3, 64)` with `DoubleConv` defined in the same file produces no node, whether called with positional or keyword arguments. This is the single reason the Hugging Face files, nanoGPT, minGPT, litgpt-style and UNet-style code parse to one or two nodes: the model body is always one level down. (A same-file class inside a `ModuleList` comprehension is expanded, which is how DiT reached 141 nodes; an imported one is mapped to an opaque block node by name, which is where MAE's 32 `transformerBlock` nodes come from. The plain attribute form gets neither.)
2. **Sub-modules built by factory or builder calls** (13 files): `create_conv2d(...)`, `nn.Sequential(*builder(...))`, `get_down_block(...)`, `build_module(...)`, `parse_model(yaml)`, `build_conv_layer(...)`, `norm_class(...)`, `conv_module(...)`, `embed_layer(...)`, `block_fn(...)`. timm, mmdetection, diffusers, Megatron, fairseq, ultralytics and YOLO all build this way. A static parser cannot run the factory, but it could register the assignment as an opaque node with the call name, so the graph keeps its skeleton.
3. **Composition through other files or through `__init__` arguments** (13 files): `Sam(image_encoder, prompt_encoder, mask_decoder)`, `UNet` from `unet_parts`, `Transformer(Encoder(...), Decoder(...))`, `ControlNet` from `ldm` helpers. Same fix as above: an opaque node per attribute rather than nothing.
4. **Containers built imperatively** (6 files): `nn.ModuleList()` followed by `.append` in a loop, `nn.Sequential(*layers)` from a list assembled in a loop, `nn.ModuleList(nn.Linear(n, k) for n, k in zip(...))`. All three return null. Surprisingly, `nn.ModuleList([nn.Linear(16, 16) for _ in range(4)])` also returns null while the same comprehension over a same-file class works; that is a bug, not a limitation.
5. **Multi-line constructor calls** (3 files as primary cause, many more as a secondary one): `self.proj = nn.Conv2d(\n    3,\n    16,\n)` is dropped. This is what black produces on any call longer than 88 characters, so it is common in exactly the codebases with the most layers (SAM, stable-diffusion, fairseq).
6. **`torch.nn.X` instead of `nn.X`** (2 files, torchaudio and Meta's llama): the fully-qualified prefix is not matched at all.
7. **Last-class heuristic**: the engine reads the last class with `__init__` and `forward`. On 12 of the 43 unparsed files the file does define the model, and the engine read something else that comes after it: a utility (`RMSNorm` in litgpt, `LayerNorm` in ConvNeXt, `NormPipe` in gpt-neox), a head or variant (`CLIPForImageClassification` over `CLIPModel`, `EfficientNetFeatures` over `EfficientNet`, `NormalizedEmbedding` over the speechbrain Transformer, `PatchEmbed` over `ImageEncoderViT`, `MLP` over `MaskDecoder`), or a wrapper (`Ensemble`, `Joiner` over `Backbone`, `ComposerMPTCausalLM` over `MPTForCausalLM`, `GaussianDiffusion` over `Unet`). On 7 of the 73 parsed files a helper defined after the model shadowed it the same way (`MLP` over DETR, `TransformerDecoderLayer` over `Transformer`, `LLaMAMLP` over TinyLlama's `GPT`, `ParallelTransformerBlock` over the `PaLM` factory, `EncoderUNetModel` over `UNetModel` in both guided-diffusion and stable-diffusion, `RobertaEncoder` over `RobertaModel`). Preferring the class with the most layer constructors, or the one no other class instantiates, would fix most of these with no new parsing.
8. **Tensor-method reshapes in `forward`** (`x.view(...)`, `x.reshape(...)`, `x.flatten(1)`) are invisible, while `torch.flatten(x, 1)` is seen. This produced the only block and three of the warnings.
9. **Functional ops nested inside a call** (`self.w_2(F.relu(self.w_1(x)))`) are found but placed after both linears, which produces `consecutive-linear-no-activation` on correct code.
10. **Parallel projections chained in init order**: `q_proj`, `k_proj`, `v_proj` are three inputs to one attention op, not a chain. The parser has no notion of two attributes consuming the same tensor, so it chains them and the linter flags them (CLIP, RWKV, two hand-written Transformers).
11. **A weight-tying assignment** (`self.transformer.wte.weight = self.lm_head.weight`) or a `.apply(...)` call on an existing attribute creates a duplicate node, which then "feeds into itself" (nanoGPT, DeiT).
12. **Positional information that is not a module**: an `nn.Parameter` added in `forward`, functional rotary embeddings, or a `pos` tensor passed in. `attention-no-pe` fired six times and was never right.
13. **A multi-line triple-quoted string inside a call in `__init__`** (`raise ValueError("""...""")`) aborts the parse of the entire file, including classes that would otherwise parse. One file (`pytorch/examples` word language model), but it is the kind of failure that silently hides a whole file.
14. **Unresolved symbols propagate as strings.** Any argument that is not a literal (`config.hidden_size`, `dims[0]`, `self.inplanes`, `width // 2`, `1 if aa_layer else 2`) is stored verbatim, then propagated against the default `[1,28,28]` input, and `invalid-output-shape` fires. 60 of 73 parsed graphs carry at least one such parameter. The rule should be suppressed when any dimension in the computed shape is non-numeric; that single change removes 54 of the 76 warnings in this study.

## How to read this as a roadmap

The items above are ordered by the number of files they would unlock, and the first, fifth, sixth, seventh and fourteenth need no new language understanding: inline same-file classes, accept multi-line calls, accept the `torch.nn.` prefix, pick the model class by content, and stop the shape rule from grading strings. Those five would move the "graph a person would recognise" count well past 9 of 116 and remove most of the 90 artefacts, and they can each be pinned by the minimal probes recorded here. The factory and composition categories (26 files) are where a static parser genuinely cannot do better than an opaque placeholder node, and the honest product answer there is the graph format itself: a `.neurarch.json` written once by the author or by an agent, rather than a parser guessing.

## Reproducing

```
node scripts/lint-real-repos.mjs --repos-dir /some/dir --clone
```

The manifest pins `owner/name` and relative file paths, not commits; the clones are whatever `HEAD` was on 2026-08-31. A later run against moved-on upstreams may differ in file contents, which is why `docs/real-repos-results.json` records the per-file line count and class inventory alongside each result. To re-judge, edit `scripts/real-repos-verdicts.json` and rerun; the summary block recomputes real/artefact/not-a-bug counts from it and never from the doc.
