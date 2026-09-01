# The static parser against the same 59 repositories, after the first round of fixes

**Date:** 2026-09-01
**Engine:** `src/vendor/engine.bundle.mjs`, first line `var __create = Object.create;`, 446797 bytes, sha256 `d9f16e085f18000b2b76f0ae5af2ee395b43181df69e3936a8a924c8e09b9a74`. Built from neurarch `feat/parser-real-repos`.
**Command:**

```
node scripts/lint-real-repos.mjs --repos-dir <dir> --out docs/real-repos-results.after.json --verdicts <causes-only.json>
```

Same manifest, same clones (the checkouts from 2026-08-31, not re-cloned), same script. The verdict file passed in carried only the 43 `@cause` attributions from the first study, so that the per-file "cause" column below still names the construct that blocked each file before; no finding verdict was merged.

**The findings in this run were not hand-judged.** The first study read every finding against the source and found precision on real code was zero (87% parser artefacts, 13% true-but-intended). Nothing here says whether that changed. The finding counts below are what the linter emitted on the graphs it now gets, and the graphs are bigger, so there are more of them. Judging them is the next job, not this one.

This run measures one thing: how much of a real file the parser now turns into a graph. The changes were the first, fourth, fifth, sixth, seventh and thirteenth items of the first study's roadmap (inline same-file classes, accept the imperative container forms, accept multi-line calls, accept the `torch.nn.` prefix, pick the model class by content, survive a triple-quoted string in `__init__`), plus source line numbers on every node. Unresolved symbols (item 14) were deliberately left alone: `config.hidden_size` is still stored as text, and there are far more of them now because the layers that carry them are now in the graph.

## Before and after

| Measure | Before (2026-08-31) | After (2026-09-01) |
|---|---|---|
| Files attempted | 116 | 116 |
| Files that returned a graph | 73 (62.9%) | **100 (86.2%)** |
| Files that returned nothing | 43 (40 `no-model`, 3 `no-model-class`) | 16 (13 `no-model`, 3 `no-model-class`) |
| Repositories with at least one file parsed | 44 of 59 | 57 of 59 |
| Layers per parsed graph (min / median / max) | 1 / 2 / 141 | 1 / **8** / 431 |
| Parsed graphs with 3 layers or fewer | 48 of 73 | 21 of 100 |
| Parsed graphs with 10 layers or more ("a graph a person would recognise") | 9 of 73 (7.8% of all files) | **47 of 100 (40.5% of all files)** |
| Nodes produced vs. `nn.*` constructor calls in the parsed files | 450 / 794 | 2946 / 1166 |
| Parsed graphs carrying at least one unresolved parameter | 60 of 73 (228 params) | 85 of 100 (2326 params) |
| Findings by severity (not hand-judged) | block 1, warn 76, info 26 | block 4, warn 179, info 190 |
| Parsed graphs with any finding | 63 of 73 | 96 of 100 |

Two of those rows need reading carefully.

**Nodes above constructor calls.** 2946 nodes from 1166 `nn.*` calls means the parser is now unrolling stacks: a `ModuleList` of twelve blocks is twelve copies of the block's layers. When the stack depth is a literal or a resolvable signature default it is the real depth. When it is `config.num_hidden_layers` or `params.n_layers`, the parser cannot know and substitutes a fixed guess of **6** (the same guess everywhere: comprehension, generator, `append` loop). So a Hugging Face `modeling_llama.py` becomes 58 nodes, which is 6 decoder layers of 9 plus embedding, rotary, norm and head; the real Llama-3-8B has 32. Anything that counts nodes or parameters on such a graph is counting the guess. `openai/CLIP` at 431 is the extreme case: a `_make_layer` helper whose block count comes from a tuple argument the parser cannot index, guessed at 6 per stage, times the residual attention stacks guessed at 6.

**Unresolved parameters went up tenfold.** 228 to 2326. This is the same class of value as before (`config.hidden_size`, `self.head_dim`, `dim * mult`), stored verbatim. The first study attributed 54 of its 76 warnings to `invalid-output-shape` firing on such a string; that rule fired 99 times here. Nothing about that changed and nothing here should be read as if it had.

## Which of the 43 previously unparsed files now parse

27 of 43. Of those 27, 15 come back with 10 or more layers. The "cause" column is the first study's attribution of why the file failed then; the "class read" column is the class the parser actually chose now, computed from the parser directly (the `mainClass` field inside `real-repos-results.after.json` is the study script's mirror of the OLD last-class rule and no longer describes what the engine reads).

| Repo | File | Cause (before) | After | Class read | Layers | Unresolved |
|---|---|---|---|---|---|---|
| Lightning-AI/litgpt | litgpt/model.py | the last class in the file is a norm or  | parsed | GPT | 39 | 66 |
| huggingface/transformers | src/transformers/models/clip/modeling_clip.py | sub-modules are instances of classes fro | parsed | CLIPVisionModel | 52 | 92 |
| huggingface/pytorch-image-models | timm/models/efficientnet.py | sub-modules built by factory or builder  | no-model |  |  |  |
| huggingface/pytorch-image-models | timm/models/swin_transformer.py | sub-modules built by factory or builder  | parsed | SwinTransformer | 42 | 12 |
| facebookresearch/dinov2 | dinov2/models/vision_transformer.py | sub-modules built by factory or builder  | no-model |  |  |  |
| facebookresearch/mae | models_vit.py | no class defines both __init__ and forwa | no-model-class |  |  |  |
| facebookresearch/detr | models/backbone.py | sub-modules are instances of classes fro | no-model |  |  |  |
| facebookresearch/segment-anything | segment_anything/modeling/image_encoder.py | constructor call split across lines (bla | parsed | ImageEncoderViT | 3 | 3 |
| facebookresearch/segment-anything | segment_anything/modeling/sam.py | sub-modules are instances of classes fro | no-model |  |  |  |
| facebookresearch/segment-anything | segment_anything/modeling/mask_decoder.py | container built imperatively (ModuleList | parsed | MaskDecoder | 25 | 21 |
| facebookresearch/ConvNeXt | models/convnext.py | the last class in the file is a norm or  | parsed | ConvNeXt | 1 | 2 |
| lucidrains/denoising-diffusion-pytorch | denoising_diffusion_pytorch/denoising_diffusion_pytorch.py | sub-modules are instances of classes fro | parsed | ResnetBlock | 12 | 10 |
| meta-llama/llama | llama/model.py | container built imperatively (ModuleList | parsed | Transformer | 69 | 0 |
| meta-llama/llama3 | llama/model.py | container built imperatively (ModuleList | parsed | Transformer | 69 | 0 |
| EleutherAI/gpt-neox | megatron/model/gpt2_model.py | no class defines both __init__ and forwa | no-model-class |  |  |  |
| EleutherAI/gpt-neox | megatron/model/transformer.py | sub-modules built by factory or builder  | parsed | ParallelTransformerLayer | 6 | 1 |
| CompVis/stable-diffusion | ldm/modules/attention.py | constructor call split across lines (bla | parsed | SpatialTransformer | 23 | 28 |
| lllyasviel/ControlNet | cldm/cldm.py | sub-modules are instances of classes fro | parsed | ControlNet | 2 | 0 |
| ultralytics/ultralytics | ultralytics/nn/tasks.py | sub-modules are instances of classes fro | no-model |  |  |  |
| ultralytics/ultralytics | ultralytics/nn/modules/block.py | sub-modules are instances of classes fro | parsed | ImagePoolingAttn | 8 | 2 |
| WongKinYiu/yolov7 | models/yolo.py | sub-modules built by factory or builder  | no-model |  |  |  |
| WongKinYiu/yolov7 | models/common.py | sub-modules are instances of classes fro | parsed | SwinTransformer2Block | 72 | 60 |
| milesial/Pytorch-UNet | unet/unet_model.py | sub-modules are instances of classes fro | no-model |  |  |  |
| pytorch/examples | word_language_model/model.py | a triple-quoted string spanning lines in | parsed | RNNModel | 5 | 9 |
| hyunwoongko/transformer | models/model/transformer.py | sub-modules are instances of classes fro | no-model |  |  |  |
| kyegomez/BitNet | bitnet/bitlinear.py | no class defines both __init__ and forwa | no-model-class |  |  |  |
| NVIDIA/Megatron-LM | megatron/core/transformer/transformer_layer.py | sub-modules built by factory or builder  | no-model |  |  |  |
| NVIDIA/Megatron-LM | megatron/core/transformer/attention.py | sub-modules built by factory or builder  | no-model |  |  |  |
| microsoft/LoRA | loralib/layers.py | sub-modules built by factory or builder  | no-model |  |  |  |
| ashawkey/stable-dreamfusion | nerf/renderer.py | sub-modules are instances of classes fro | no-model |  |  |  |
| huggingface/diffusers | src/diffusers/models/attention.py | container built imperatively (ModuleList | parsed | BasicTransformerBlock | 14 | 6 |
| huggingface/diffusers | src/diffusers/models/unets/unet_2d_condition.py | sub-modules built by factory or builder  | parsed | UNet2DConditionModel | 4 | 10 |
| open-mmlab/mmdetection | mmdet/models/backbones/resnet.py | sub-modules built by factory or builder  | parsed | Bottleneck | 1 | 0 |
| pytorch/audio | src/torchaudio/models/wav2vec2/model.py | sub-modules are instances of classes fro | no-model |  |  |  |
| pytorch/audio | src/torchaudio/models/wav2vec2/components.py | torch.nn.X fully-qualified prefix instea | parsed | EncoderLayer | 5 | 5 |
| pytorch/audio | src/torchaudio/models/conformer.py | torch.nn.X fully-qualified prefix instea | parsed | Conformer | 96 | 84 |
| speechbrain/speechbrain | speechbrain/lobes/models/transformer/Transformer.py | sub-modules built by factory or builder  | parsed | TransformerInterface | 39 | 0 |
| facebookresearch/fairseq | fairseq/models/transformer/transformer_encoder.py | sub-modules built by factory or builder  | parsed | TransformerEncoderBase | 4 | 2 |
| facebookresearch/fairseq | fairseq/modules/multihead_attention.py | constructor call split across lines (bla | parsed | MultiheadAttention | 11 | 20 |
| ultralytics/yolov5 | models/yolo.py | sub-modules built by factory or builder  | parsed | Segment | 1 | 2 |
| mosaicml/llm-foundry | llmfoundry/models/mpt/modeling_mpt.py | sub-modules are instances of classes fro | parsed | MPTModel | 4 | 4 |
| eriklindernoren/PyTorch-GAN | implementations/dcgan/dcgan.py | container built imperatively (ModuleList | parsed | Generator | 12 | 3 |
| eriklindernoren/PyTorch-GAN | implementations/cyclegan/models.py | container built imperatively (ModuleList | parsed | GeneratorResNet | 32 | 36 |

## The 73 files that already parsed: what the same files yield now

"Class before" is the class the old rule read (last with `__init__` and `forward`); "class after" is the root the new rule chose. A changed class is not necessarily an improvement (see item 4 below).

| Repo | File | Class before | Class after | Layers before | Layers after | Unresolved before | Unresolved after |
|---|---|---|---|---|---|---|---|
| karpathy/nanoGPT | model.py | GPT | GPT | 2 | 72 | 4 | 75 |
| karpathy/minGPT | mingpt/model.py | GPT | GPT | 1 | 47 | 2 | 56 |
| karpathy/nanochat | nanochat/gpt.py | GPT | GPT | 2 | 63 | 0 | 2 |
| huggingface/transformers | src/transformers/models/llama/modeling_llama.py | LlamaForCausalLM | LlamaForCausalLM | 1 | 58 | 2 | 88 |
| huggingface/transformers | src/transformers/models/bert/modeling_bert.py | BertForQuestionAnswering | BertForPreTraining | 2 | 23 | 2 | 16 |
| huggingface/transformers | src/transformers/models/gpt2/modeling_gpt2.py | GPT2ForQuestionAnswering | GPT2DoubleHeadsModel | 5 | 44 | 8 | 48 |
| huggingface/transformers | src/transformers/models/vit/modeling_vit.py | ViTForImageClassification | ViTForImageClassification | 1 | 59 | 2 | 100 |
| huggingface/transformers | src/transformers/models/whisper/modeling_whisper.py | WhisperForAudioClassification | WhisperForConditionalGeneration | 3 | 135 | 3 | 234 |
| huggingface/transformers | src/transformers/models/qwen2/modeling_qwen2.py | Qwen2ForCausalLM | Qwen2ForCausalLM | 1 | 58 | 2 | 88 |
| huggingface/transformers | src/transformers/models/mistral/modeling_mistral.py | MistralForCausalLM | MistralForCausalLM | 1 | 58 | 2 | 88 |
| huggingface/transformers | src/transformers/models/mixtral/modeling_mixtral.py | MixtralForCausalLM | MixtralForCausalLM | 1 | 46 | 2 | 52 |
| huggingface/transformers | src/transformers/models/gemma/modeling_gemma.py | GemmaForCausalLM | GemmaForCausalLM | 1 | 58 | 2 | 86 |
| huggingface/pytorch-image-models | timm/models/vision_transformer.py | VisionTransformer | DiffParallelScalingBlock | 3 | 7 | 4 | 6 |
| huggingface/pytorch-image-models | timm/models/resnet.py | ResNet | ResNet | 5 | 7 | 6 | 9 |
| huggingface/pytorch-image-models | timm/models/convnext.py | ConvNeXt | ConvNeXt | 2 | 3 | 6 | 4 |
| pytorch/vision | torchvision/models/resnet.py | ResNet | ResNet | 5 | 5 | 3 | 3 |
| pytorch/vision | torchvision/models/vgg.py | VGG | VGG | 3 | 9 | 1 | 1 |
| pytorch/vision | torchvision/models/densenet.py | DenseNet | DenseNet | 5 | 8 | 3 | 2 |
| pytorch/vision | torchvision/models/mobilenetv2.py | MobileNetV2 | MobileNetV2 | 1 | 2 | 1 | 1 |
| pytorch/vision | torchvision/models/vision_transformer.py | VisionTransformer | VisionTransformer | 1 | 1 | 1 | 0 |
| facebookresearch/DiT | models.py | DiT | DiT | 141 | 177 | 0 | 2 |
| facebookresearch/mae | models_mae.py | MaskedAutoencoderViT | MaskedAutoencoderViT | 35 | 35 | 4 | 4 |
| facebookresearch/detr | models/detr.py | MLP | DETR | 1 | 4 | 0 | 6 |
| facebookresearch/detr | models/transformer.py | TransformerDecoderLayer | TransformerDecoderLayer | 11 | 11 | 15 | 15 |
| facebookresearch/deit | models.py | DistilledVisionTransformer | DistilledVisionTransformer | 2 | 2 | 4 | 4 |
| lucidrains/vit-pytorch | vit_pytorch/vit.py | ViT | ViT | 4 | 7 | 3 | 8 |
| lucidrains/vit-pytorch | vit_pytorch/simple_vit.py | SimpleViT | SimpleViT | 2 | 5 | 2 | 7 |
| lucidrains/x-transformers | x_transformers/x_transformers.py | XTransformer | XTransformer | 14 | 18 | 6 | 16 |
| lucidrains/PaLM-pytorch | palm_pytorch/palm_pytorch.py | ParallelTransformerBlock | ParallelTransformerBlock | 4 | 7 | 4 | 6 |
| openai/whisper | whisper/model.py | Whisper | Whisper | 3 | 4 | 2 | 2 |
| openai/CLIP | clip/model.py | CLIP | CLIP | 25 | 431 | 13 | 176 |
| openai/guided-diffusion | guided_diffusion/unet.py | EncoderUNetModel | EncoderUNetModel | 1 | 5 | 1 | 0 |
| state-spaces/mamba | mamba_ssm/modules/mamba_simple.py | Mamba | Mamba | 12 | 14 | 2 | 9 |
| state-spaces/mamba | mamba_ssm/models/mixer_seq_simple.py | MambaLMHeadModel | MambaLMHeadModel | 1 | 2 | 2 | 4 |
| BlinkDL/RWKV-LM | RWKV-v4neo/src/model.py | RWKV | RWKV | 14 | 212 | 14 | 302 |
| BlinkDL/RWKV-LM | RWKV-v7/rwkv_v7_demo.py | RWKV | RWKV | 9 | 81 | 5 | 107 |
| deepseek-ai/DeepSeek-V3 | inference/model.py | Transformer | Transformer | 2 | 87 | 0 | 0 |
| mistralai/mistral-inference | src/mistral_inference/transformer.py | Transformer | Transformer | 1 | 1 | 2 | 2 |
| jzhang38/TinyLlama | lit_gpt/model.py | LLaMAMLP | GPT | 1 | 14 | 0 | 28 |
| CompVis/stable-diffusion | ldm/modules/diffusionmodules/openaimodel.py | EncoderUNetModel | EncoderUNetModel | 1 | 5 | 1 | 0 |
| milesial/Pytorch-UNet | unet/unet_parts.py | OutConv | Up | 1 | 8 | 2 | 4 |
| kuangliu/pytorch-cifar | models/resnet.py | ResNet | Bottleneck | 4 | 11 | 2 | 9 |
| kuangliu/pytorch-cifar | models/vgg.py | VGG | VGG | 6 | 6 | 2 | 2 |
| kuangliu/pytorch-cifar | models/densenet.py | DenseNet | DenseNet | 5 | 17 | 3 | 9 |
| kuangliu/pytorch-cifar | models/mobilenetv2.py | MobileNetV2 | MobileNetV2 | 9 | 17 | 1 | 9 |
| pytorch/examples | mnist/main.py | Net | Net | 11 | 11 | 0 | 0 |
| jadore801120/attention-is-all-you-need-pytorch | transformer/Models.py | Transformer | Transformer | 1 | 21 | 2 | 4 |
| jadore801120/attention-is-all-you-need-pytorch | transformer/SubLayers.py | PositionwiseFeedForward | MultiHeadAttention | 5 | 7 | 6 | 10 |
| jadore801120/attention-is-all-you-need-pytorch | transformer/Layers.py | DecoderLayer | DecoderLayer | 2 | 3 | 0 | 0 |
| hyunwoongko/transformer | models/model/encoder.py | Encoder | Encoder | 6 | 7 | 0 | 0 |
| hyunwoongko/transformer | models/blocks/encoder_layer.py | EncoderLayer | EncoderLayer | 3 | 6 | 2 | 2 |
| hyunwoongko/transformer | models/layers/multi_head_attention.py | MultiHeadAttention | MultiHeadAttention | 4 | 5 | 8 | 8 |
| kyegomez/BitNet | bitnet/bit_transformer.py | BitNetTransformer | BitNetTransformer | 3 | 3 | 3 | 4 |
| Dao-AILab/flash-attention | flash_attn/models/gpt.py | GPTLMHeadModel | GPTLMHeadModel | 3 | 4 | 4 | 5 |
| Dao-AILab/flash-attention | flash_attn/modules/mha.py | ParallelMHA | MHA | 8 | 20 | 0 | 19 |
| NVIDIA/Megatron-LM | megatron/core/models/gpt/gpt_model.py | GPTModel | GPTModel | 1 | 1 | 0 | 0 |
| ashawkey/stable-dreamfusion | nerf/network.py | NeRFNetwork | ResBlock | 1 | 4 | 0 | 5 |
| huggingface/diffusers | src/diffusers/models/unets/unet_2d.py | UNet2DModel | UNet2DModel | 7 | 8 | 13 | 12 |
| open-mmlab/mmdetection | mmdet/models/backbones/swin.py | SwinTransformer | SwinTransformer | 2 | 2 | 1 | 1 |
| speechbrain/speechbrain | speechbrain/nnet/attention.py | RoPEMHA | LocationAwareAttention | 2 | 8 | 2 | 12 |
| rwightman/gen-efficientnet-pytorch | geffnet/gen_efficientnet.py | GenEfficientNet | GenEfficientNet | 3 | 3 | 2 | 2 |
| rwightman/gen-efficientnet-pytorch | geffnet/mobilenetv3.py | MobileNetV3 | MobileNetV3 | 2 | 2 | 2 | 2 |
| xinntao/ESRGAN | RRDBNet_arch.py | RRDBNet | RRDB | 11 | 18 | 12 | 9 |
| NVlabs/stylegan2-ada-pytorch | training/networks.py | Discriminator | Discriminator | 2 | 6 | 0 | 0 |
| facebookresearch/fairseq | fairseq/models/roberta/model.py | RobertaEncoder | RobertaClassificationHead | 1 | 4 | 2 | 6 |
| ultralytics/yolov5 | models/common.py | Classify | BottleneckCSP | 3 | 12 | 3 | 16 |
| microsoft/Swin-Transformer | models/swin_transformer.py | SwinTransformer | SwinTransformer | 1 | 1 | 2 | 2 |
| google/gemma_pytorch | gemma/model.py | GemmaForCausalLM | GemmaForCausalLM | 2 | 158 | 0 | 0 |
| black-forest-labs/flux | src/flux/model.py | Flux | Flux | 2 | 2 | 4 | 4 |
| black-forest-labs/flux | src/flux/modules/layers.py | LastLayer | DoubleStreamBlock | 2 | 22 | 3 | 24 |
| yunjey/pytorch-tutorial | tutorials/02-intermediate/convolutional_neural_network/main.py | ConvNet | ConvNet | 3 | 9 | 2 | 2 |
| yunjey/pytorch-tutorial | tutorials/03-advanced/image_captioning/model.py | DecoderRNN | DecoderRNN | 3 | 3 | 7 | 7 |
| microsoft/unilm | beit/modeling_finetune.py | VisionTransformer | VisionTransformer | 1 | 1 | 2 | 2 |

## Findings by rule, before and after (not judged)

| Rule | Severity | Before | After |
|---|---|---|---|
| consecutive-linear-no-activation | info | 14 | 103 |
| invalid-output-shape | warn | 54 | 99 |
| deep-no-residual | warn | 0 | 30 |
| double-norm | info | 2 | 23 |
| dropout-before-bn | info | 2 | 21 |
| deep-no-norm | info | 6 | 18 |
| vanishing-gradient | info | 0 | 12 |
| attention-no-pe | warn | 6 | 11 |
| bn-at-output | warn | 4 | 11 |
| init-activation-mismatch | info | 0 | 8 |
| redundant-activation | warn | 2 | 7 |
| dropout-at-output | warn | 5 | 6 |
| bn-after-activation | warn | 1 | 5 |
| duplicate-positional-encoding | info | 1 | 4 |
| linear-after-conv-no-flatten | block | 1 | 4 |
| pool-into-linear-no-flatten | warn | 3 | 3 |
| deep-attention-default-init | info | 1 | 2 |
| non-spatial-into-conv | warn | 0 | 2 |
| flatten-into-attention | warn | 0 | 2 |
| output-activation | info | 1 | 1 |
| moe-no-aux-loss | info | 0 | 1 |

## The 16 files that still return nothing

| Repo | File | Why (first study's attribution, still accurate) |
|---|---|---|
| huggingface/pytorch-image-models | timm/models/efficientnet.py | blocks built by `builder(...)` |
| facebookresearch/dinov2 | dinov2/models/vision_transformer.py | `block_fn(...)`, `embed_layer(...)` factories |
| facebookresearch/mae | models_vit.py | subclass of timm's VisionTransformer; no `__init__` layers of its own |
| facebookresearch/detr | models/backbone.py | torchvision backbone passed in |
| facebookresearch/segment-anything | segment_anything/modeling/sam.py | `Sam(image_encoder, prompt_encoder, mask_decoder)`: all three passed in |
| EleutherAI/gpt-neox | megatron/model/gpt2_model.py | pipeline module, no `forward` |
| ultralytics/ultralytics | ultralytics/nn/tasks.py | `parse_model(yaml)` |
| WongKinYiu/yolov7 | models/yolo.py | `parse_model(yaml)` |
| milesial/Pytorch-UNet | unet/unet_model.py | `DoubleConv`, `Down`, `Up`, `OutConv` imported from `unet_parts` |
| hyunwoongko/transformer | models/model/transformer.py | `Encoder`, `Decoder` imported |
| kyegomez/BitNet | bitnet/bitlinear.py | a single `nn.Linear` subclass overriding `forward` |
| NVIDIA/Megatron-LM | megatron/core/transformer/transformer_layer.py | `build_module(spec)` |
| NVIDIA/Megatron-LM | megatron/core/transformer/attention.py | `build_module(spec)` |
| ashawkey/stable-dreamfusion | nerf/renderer.py | sub-modules from other files |
| pytorch/audio | src/torchaudio/models/wav2vec2/model.py | `Wav2Vec2Model(feature_extractor, encoder)`: both passed in |
| microsoft/LoRA | loralib/layers.py | mixins over `nn.Linear` / `nn.Conv2d`; the only class with layers of its own has no `forward` (this file returned a 1-node graph in the first study, from that class) |

Every one of these is composition through another file, a builder call, or a
class that owns no layers (or, for LoRA, no forward). A static parser reads none of them without an
opaque placeholder node per unknown attribute, which the first study's third
roadmap item asks for and this round did not add: the component registry has
no type for "a module we could not read", and inventing one changes what the
shape rules see. That is the next parser decision, and it is a product
decision as much as a parser one.

## What changed in the parser, in the order the study ranked it

1. **Same-file classes are inlined** (13 files were attributed to this, and
   it is also why every Hugging Face file was one node). `self.model =
   LlamaModel(config)` now expands `LlamaModel`, whose `layers` expand
   `LlamaDecoderLayer`, whose `self_attn` expands `LlamaAttention`, four
   levels, which is exactly the depth cap. Nodes carry the state_dict path
   (`model.layers.0.self_attn.q_proj`) as scope plus name, and `sourceLine` /
   `sourceClass` point at the line and class where the layer was written, not
   the instantiation site. A class whose `__init__` holds only parameters
   (`LlamaRMSNorm`, gemma's `Linear`, StyleGAN's `FullyConnectedLayer`) expands
   to nothing and instead maps through the known-class table by name, with a
   suffix match so `Qwen2RMSNorm` reaches `rmsnorm`.
2. **Containers built imperatively** (6 files). `nn.ModuleList([nn.Linear(16,
   16) for _ in range(4)])` was a plain bug (the comprehension regex did not
   admit the `nn.` prefix) and is fixed; the generator form, `ModuleList()`
   followed by `.append` in a loop, a local list spread into
   `nn.Sequential(*layers)`, a starred comprehension, and `nn.ModuleDict`
   entries are all read. `meta-llama/llama` goes from null to 69 nodes on this.
3. **Multi-line constructors and `torch.nn.X`** (3 + 2 files). `__init__` is
   now walked as logical statements from the statement tree, so a call black
   split across lines is one statement with one line number. `torch.nn.` is
   normalised to `nn.` before any regex looks. The argument extractor is
   paren-balanced, so `kernel_size=(3, 3)` no longer truncates the argument
   list. `torchaudio/conformer.py` goes from null to 96 nodes.
4. **Root class selection** (12 unparsed and 7 misread files). The class read
   is now, among classes no other class in the file instantiates, the one with
   the most layers after inlining; ties prefer a class with `forward`, then a
   name matching the file stem, then a name ending in `Model` or
   `ForCausalLM`, then the last in the file. `graphFromPyTorchSource(code,
   name, { className })` requests a specific class. This is what moved
   `BertForQuestionAnswering` to `BertForPreTraining`, `LLaMAMLP` to `GPT`
   (TinyLlama), and `MLP` to `DETR`. It has a known failure: when the real
   model builds its blocks through a factory or a class passed as an argument,
   it has few layers of its own and a helper block wins on count.
   `kuangliu/pytorch-cifar/models/resnet.py` is read as `Bottleneck` (11
   nodes) rather than `ResNet` (4), and timm's `vision_transformer.py` as
   `DiffParallelScalingBlock` rather than `VisionTransformer`. The rule is
   honest about what it can see and wrong about what the file is for; a
   `className` option exists for callers that know better.
5. **A triple-quoted string inside a call in `__init__`** (1 file, but the
   kind that silently hides a whole file). The logical-line tokenizer reset its
   string state on every physical line, so a `)` inside the second line of a
   `raise ValueError("""...""")` closed the call and the rest of the file
   became one statement. String state now carries across lines.
   `pytorch/examples/word_language_model` goes from null to 5 nodes.
6. **Forward references by index.** Once stacks were unrolled, a second drop
   showed up: `for block_idx, block in enumerate(self.transformer.h)` and
   `layer = self.layers[i]` were not recognised as calls, and because
   `forward()` order drives the graph, the whole stack vanished (litgpt read 3
   nodes with its 36-node stack silently dropped; gemma read 1). A `self.X`
   reference inside a loop iterable or a subscript now counts as one call to
   the container.

Not changed, on purpose: `config.hidden_size` and every other non-literal
argument is still stored as text (roadmap item 14, the `invalid-output-shape`
suppression, is a linter change and belongs with the finding re-judgement);
tensor-method reshapes in `forward` are still invisible; parallel projections
are still chained.

## On the four blocks

All four are `linear-after-conv-no-flatten`. They were not judged, but the
source between the two layers is worth recording for whoever does: in
`detr.py` the transformer sits between `input_proj` and `class_embed` and is
built by an imported `build_transformer`, so it is missing from the graph; in
`mask_decoder.py` the upscaling convolution and the hypernetwork MLPs are
parallel branches that a chain cannot express; in speechbrain's
`attention.py` a `.transpose` and in StyleGAN's `networks.py` a
`.flatten(1)` sit between the two, both tensor methods the parser does not
see. The first study's one block had the same shape. This rule has a real
crash measurement behind it on graphs the app authored; on graphs the parser
reconstructed from a chain of `self.x(...)` calls it is grading the
reconstruction.

## Reproducing

```
node scripts/lint-real-repos.mjs --repos-dir <dir> --out docs/real-repos-results.after.json --verdicts <causes-only.json>
```

with `<causes-only.json>` holding only the `|@cause` entries of
`scripts/real-repos-verdicts.json`. The class the parser actually read is
not in the results file (its `mainClass` mirrors the old rule); it is the
`sourceClass` on the graph's unscoped components, or `parsePyTorchCode(code,
{ fileName }).mainClass` in the app repo. The previous results and the
hand-judged findings are unchanged in `docs/real-repos-results.json` and
`docs/REAL_REPOS_STUDY.md`.
