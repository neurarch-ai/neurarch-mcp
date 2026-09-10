# The static parser and linter against 59 real PyTorch repositories, with edge provenance

**Date:** 2026-09-09. Supersedes [REAL_REPOS_STUDY-2026-09-03.md](./REAL_REPOS_STUDY-2026-09-03.md).
**Engine:** `src/vendor/engine.bundle.mjs` rebuilt from neurarch `main`, sha256 `3c59d00daf91`, carrying per-edge inference provenance.
**Command:** `npx tsx scripts/lint-real-repos.mjs --repos-dir <dir> --clone`, same 59 repositories and 116 files as the last three runs (`scripts/real-repos.json`).

The last study named the change to make and this is the measurement of it. Nothing about the parser's reach moved: the same 100 files parse, the same 47 are substantial, the same 2946 nodes come back. What changed is that the importer now says WHY each edge exists, and the rules that are about adjacency refuse to speak for an edge nobody wrote.

## The change

A `.py` is not a graph. The importer reads `forward()` and chains the calls it recognises, which is right for

```python
h = self.a(x)
y = self.b(h)
```

and wrong for

```python
q = self.head_q(x)
k = self.head_k(x)
```

where the two projections are parallel branches off one input and the edge between them exists only because the two statements are adjacent in the file. Every edge now carries `inferred` when the source did not show the value flowing:

- `construction-order` — no variable carries the first statement's output into the second, or there was no readable `forward()` at all and the chain is `__init__` order.
- `unmodelled-merge` — the statement combined the value with an earlier one (`x = x + self.attn(...)`, or HuggingFace's `hidden_states = residual + hidden_states`) and the graph has no node for that add.

Rules whose whole claim is about an edge (`bn-after-activation`, `dropout-before-bn`, `redundant-activation`, `consecutive-linear-no-activation`, `double-norm`, the flatten family, the three at-output rules, `init-activation-mismatch`) now iterate only observed edges. `deep-no-residual` holds back on any graph carrying an `unmodelled-merge`, because "no skip connections" is a statement about the parse when the source showed us an add we failed to draw. `deep-no-norm` ends a conv run at an inferred edge. Propagated-shape findings (`invalid-output-shape`, `merge-shape-mismatch`, `attention-in-mismatch`, `compute-error`) are dropped downstream of an inferred edge; head divisibility and unknown layer types are unaffected, because those are facts about one layer's own parameters.

Rules about a layer's own parameters do not consult provenance at all. `embed_dim % num_heads` is wrong whatever feeds it.

## Headline numbers

| Measure | 2026-09-03 | 2026-09-09 |
|---|---|---|
| Files that returned a graph | 100 of 116 (86.2%) | 100 of 116 (86.2%) |
| Layers per parsed graph, median | 8 | 8 |
| Parsed graphs with 10 layers or more | 47 | 47 |
| Findings shown (block / warn / info) | 4 / 88 / 190 | **3 / 34 / 83** |
| Parsed graphs with a block | 4 | 3 |

## Hand-judged precision

Verdicts are unchanged in `scripts/real-repos-verdicts.json`; what moved is which findings still fire.

| Severity | 2026-09-03 total / artefact / not-a-bug | 2026-09-09 total / artefact / not-a-bug |
|---|---|---|
| block | 4 / 3 / 1 | 3 / 2 / 1 |
| warn | 88 / 70 / 18 | 34 / 22 / 12 |
| **block + warn** | **92 / 73 / 19** | **37 / 24 / 13** |

**49 of the 73 artefacts are gone (67%), and 13 of the 19 true-but-not-a-bug findings are kept.** Real defects found: still zero, on either engine. This filter cannot create precision the parser does not have; it removes findings that were never about the code.

The four biggest movers are the four the last study predicted. `deep-no-residual` 30 to 4: llama, Qwen2, Mistral, Gemma, ViT, CLIP, TinyLlama, conformer and the rest write their residual as an add the graph has no node for, and now say so. `dropout-at-output` 6 to 1 and `bn-at-output` 11 to 6: the Output node is synthetic, hung off the last statement the importer recognised, and it is now marked inferred unless `forward()` actually returns that statement's value (read across line continuations, because fairseq and HuggingFace return a multi-line dict). `invalid-output-shape` 8 to 2. `consecutive-linear-no-activation`, an info rule and the loudest one in the file, 103 to 27.

### The six true findings that were lost

`bn-at-output` on BitNet, conformer and CLIP, `bn-after-activation` on RWKV-7, `deep-no-norm` on ESRGAN, `redundant-activation` on gemma\_pytorch. Each is a true statement about a partial view (a pre-norm encoder's output really is normalized hidden states; ESRGAN really does drop BatchNorm on purpose), and each is lost because the file around it parses thinly enough that the edge in question was inferred. They are the cost of the filter and they are recorded here rather than rounded away.

### The 24 artefacts that remain

Three causes, none of them an edge the filter can see:

- **A functional op the registry does not map** (3 `attention-no-pe`: timm, DiT and BERT apply RoPE or add a position embedding through a call the parser does not turn into a node). This is a dispatch-table gap, not a topology one.
- **A parse thin enough that the whole file is one fragment** (`pool-into-linear-no-flatten` on the three CIFAR models, `linear-after-conv-no-flatten` on DETR and StyleGAN2). The two layers really are adjacent in what was parsed; what is missing is the rest of the model.
- **A container the importer unrolled once** (`flatten-into-attention` on the two UNets, `deep-no-residual` on the four remaining files, where the residual is in a class that was not inlined).

## What this means for the package

- **`trace_model` is still the path for real code.** The parser is now honest about what it inferred instead of quiet about it, which is a different thing from being right.
- **A quiet file is no longer the same as a checked file, and the tools say so.** `edgeProvenance(model)` is exported from the engine; the VS Code extension renders it as one Information line per file, and `lint_model` should carry it into its response the way it already carries the dimension filter's count.
- Quote this as: the parser reads about 41% of real model files into a recognisable graph, and on those graphs it now reports 37 block+warn findings instead of 92, with the same zero real defects and two thirds of the artefacts gone. Then say `trace_model`.

## Per-repo table

`Layers` is the node count of the returned graph. `nn.* calls` is the number of `nn.<Layer>(` constructor calls in the whole file. `Unresolved` is the count of parameters stored as source text. `Verdicts` merges `scripts/real-repos-verdicts.json`.

| Repo | File | Status | Main class | Layers | nn.* calls | Unresolved | Findings | Verdicts |
|---|---|---|---|---|---|---|---|---|
| karpathy/nanoGPT | model.py | parsed | GPT | 72 | 12 | 75 | dropout-before-bn |  |
| karpathy/minGPT | mingpt/model.py | parsed | GPT | 47 | 14 | 56 | dropout-before-bn, double-norm |  |
| karpathy/nanochat | nanochat/gpt.py | parsed | GPT | 63 | 2 | 2 | vanishing-gradient x2, deep-no-norm, init-activation-mismatch |  |
| Lightning-AI/litgpt | litgpt/model.py | parsed | RMSNorm | 39 | 15 | 66 | deep-no-residual, deep-no-norm, consecutive-linear-no-activation x3 | artefact=1 |
| huggingface/transformers | src/transformers/models/llama/modeling_llama.py | parsed | LlamaForCausalLM | 58 | 11 | 88 | consecutive-linear-no-activation |  |
| huggingface/transformers | src/transformers/models/bert/modeling_bert.py | parsed | BertForQuestionAnswering | 23 | 36 | 16 | attention-no-pe, vanishing-gradient, init-activation-mismatch, deep-attention-default-init | artefact=1 |
| huggingface/transformers | src/transformers/models/gpt2/modeling_gpt2.py | parsed | GPT2ForQuestionAnswering | 44 | 19 | 48 | dropout-at-output | artefact=1 |
| huggingface/transformers | src/transformers/models/vit/modeling_vit.py | parsed | ViTForImageClassification | 59 | 16 | 100 | dropout-before-bn, consecutive-linear-no-activation x2 |  |
| huggingface/transformers | src/transformers/models/whisper/modeling_whisper.py | parsed | WhisperForAudioClassification | 135 | 23 | 234 | consecutive-linear-no-activation |  |
| huggingface/transformers | src/transformers/models/qwen2/modeling_qwen2.py | parsed | Qwen2ForCausalLM | 58 | 11 | 88 | consecutive-linear-no-activation |  |
| huggingface/transformers | src/transformers/models/mistral/modeling_mistral.py | parsed | MistralForCausalLM | 58 | 11 | 88 | consecutive-linear-no-activation |  |
| huggingface/transformers | src/transformers/models/mixtral/modeling_mixtral.py | parsed | MixtralForCausalLM | 46 | 8 | 52 | moe-no-aux-loss |  |
| huggingface/transformers | src/transformers/models/gemma/modeling_gemma.py | parsed | GemmaForCausalLM | 58 | 11 | 86 | consecutive-linear-no-activation |  |
| huggingface/transformers | src/transformers/models/clip/modeling_clip.py | parsed | CLIPForImageClassification | 52 | 22 | 92 | bn-at-output, consecutive-linear-no-activation, double-norm | not-a-bug=1 |
| huggingface/pytorch-image-models | timm/models/vision_transformer.py | parsed | VisionTransformer | 7 | 14 | 6 | attention-no-pe | artefact=1 |
| huggingface/pytorch-image-models | timm/models/resnet.py | parsed | ResNet | 7 | 14 | 9 | deep-no-norm |  |
| huggingface/pytorch-image-models | timm/models/efficientnet.py | no-model | EfficientNetFeatures |  | 1 |  | main class EfficientNetFeatures (line 356) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| huggingface/pytorch-image-models | timm/models/convnext.py | parsed | ConvNeXt | 3 | 3 | 4 | clean |  |
| huggingface/pytorch-image-models | timm/models/swin_transformer.py | parsed | SwinTransformer | 42 | 6 | 12 | deep-no-norm |  |
| pytorch/vision | torchvision/models/resnet.py | parsed | ResNet | 5 | 9 | 3 | clean |  |
| pytorch/vision | torchvision/models/vgg.py | parsed | VGG | 9 | 13 | 1 | deep-no-norm |  |
| pytorch/vision | torchvision/models/densenet.py | parsed | DenseNet | 8 | 16 | 2 | clean |  |
| pytorch/vision | torchvision/models/mobilenetv2.py | parsed | MobileNetV2 | 2 | 3 | 1 | clean |  |
| pytorch/vision | torchvision/models/vision_transformer.py | parsed | VisionTransformer | 1 | 10 | 0 | clean |  |
| facebookresearch/DiT | models.py | parsed | DiT | 177 | 14 | 2 | attention-no-pe, deep-attention-default-init | artefact=1 not-a-bug=1 |
| facebookresearch/dinov2 | dinov2/models/vision_transformer.py | no-model | DinoVisionTransformer |  | 0 |  | main class DinoVisionTransformer (line 45) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| facebookresearch/mae | models_mae.py | parsed | MaskedAutoencoderViT | 35 | 3 | 4 | deep-no-norm | artefact=1 |
| facebookresearch/mae | models_vit.py | no-model-class |  |  | 0 |  | no class defines both __init__ and forward | no class defines both __init__ and forward (subclass overriding forward only, or pipeline module) |
| facebookresearch/detr | models/detr.py | parsed | MLP | 4 | 4 | 6 | linear-after-conv-no-flatten (block) | artefact=1 |
| facebookresearch/detr | models/transformer.py | parsed | TransformerDecoderLayer | 11 | 21 | 15 | attention-no-pe | not-a-bug=1 |
| facebookresearch/detr | models/backbone.py | no-model | Joiner |  | 0 |  | main class Joiner (line 96) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| facebookresearch/segment-anything | segment_anything/modeling/image_encoder.py | parsed | PatchEmbed | 3 | 5 | 3 | clean |  |
| facebookresearch/segment-anything | segment_anything/modeling/sam.py | no-model | Sam |  | 0 |  | main class Sam (line 18) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| facebookresearch/segment-anything | segment_anything/modeling/mask_decoder.py | parsed | MLP | 25 | 5 | 21 | deep-no-residual, vanishing-gradient, deep-no-norm | artefact=1 |
| facebookresearch/ConvNeXt | models/convnext.py | parsed | LayerNorm | 1 | 8 | 2 | clean |  |
| facebookresearch/deit | models.py | parsed | DistilledVisionTransformer | 2 | 1 | 4 | clean |  |
| lucidrains/vit-pytorch | vit_pytorch/vit.py | parsed | ViT | 7 | 18 | 8 | dropout-before-bn |  |
| lucidrains/vit-pytorch | vit_pytorch/simple_vit.py | parsed | SimpleViT | 5 | 13 | 7 | double-norm |  |
| lucidrains/x-transformers | x_transformers/x_transformers.py | parsed | XTransformer | 18 | 49 | 16 | clean |  |
| lucidrains/denoising-diffusion-pytorch | denoising_diffusion_pytorch/denoising_diffusion_pytorch.py | parsed | GaussianDiffusion | 12 | 20 | 10 | clean |  |
| lucidrains/PaLM-pytorch | palm_pytorch/palm_pytorch.py | parsed | ParallelTransformerBlock | 7 | 5 | 6 | clean |  |
| openai/whisper | whisper/model.py | parsed | Whisper | 4 | 2 | 2 | bn-after-activation, bn-at-output | artefact=2 |
| openai/CLIP | clip/model.py | parsed | CLIP | 431 | 32 | 176 | clean |  |
| openai/guided-diffusion | guided_diffusion/unet.py | parsed | EncoderUNetModel | 5 | 18 | 0 | attention-no-pe, flatten-into-attention | artefact=1 not-a-bug=1 |
| state-spaces/mamba | mamba_ssm/modules/mamba_simple.py | parsed | Mamba | 14 | 6 | 9 | deep-no-norm | not-a-bug=1 |
| state-spaces/mamba | mamba_ssm/models/mixer_seq_simple.py | parsed | MambaLMHeadModel | 2 | 2 | 4 | clean |  |
| BlinkDL/RWKV-LM | RWKV-v4neo/src/model.py | parsed | RWKV | 212 | 46 | 302 | dropout-before-bn x2, vanishing-gradient, consecutive-linear-no-activation x5, init-activation-mismatch | artefact=3 |
| BlinkDL/RWKV-LM | RWKV-v7/rwkv_v7_demo.py | parsed | RWKV | 81 | 15 | 107 | vanishing-gradient x2, double-norm |  |
| deepseek-ai/DeepSeek-V3 | inference/model.py | parsed | Transformer | 87 | 0 | 0 | bn-after-activation, consecutive-linear-no-activation x3 | artefact=1 |
| mistralai/mistral-inference | src/mistral_inference/transformer.py | parsed | Transformer | 1 | 2 | 2 | clean |  |
| meta-llama/llama | llama/model.py | parsed | Transformer | 69 | 0 | 0 | consecutive-linear-no-activation |  |
| meta-llama/llama3 | llama/model.py | parsed | Transformer | 69 | 0 | 0 | consecutive-linear-no-activation |  |
| EleutherAI/gpt-neox | megatron/model/gpt2_model.py | no-model-class |  |  | 0 |  | no class defines both __init__ and forward | no class defines both __init__ and forward (subclass overriding forward only, or pipeline module) |
| EleutherAI/gpt-neox | megatron/model/transformer.py | parsed | NormPipe | 6 | 1 | 1 | invalid-output-shape | artefact=1 |
| jzhang38/TinyLlama | lit_gpt/model.py | parsed | LLaMAMLP | 14 | 9 | 28 | deep-no-norm, consecutive-linear-no-activation |  |
| CompVis/stable-diffusion | ldm/modules/diffusionmodules/openaimodel.py | parsed | EncoderUNetModel | 5 | 20 | 0 | attention-no-pe, flatten-into-attention | artefact=1 not-a-bug=1 |
| CompVis/stable-diffusion | ldm/modules/attention.py | parsed | SpatialTransformer | 23 | 22 | 28 | deep-no-residual | artefact=1 |
| lllyasviel/ControlNet | cldm/cldm.py | parsed | ControlNet | 2 | 8 | 0 | redundant-activation | artefact=1 |
| ultralytics/ultralytics | ultralytics/nn/tasks.py | no-model | Ensemble |  | 7 |  | main class Ensemble (line 1520) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| ultralytics/ultralytics | ultralytics/nn/modules/block.py | parsed | Proto26 | 8 | 53 | 2 | clean |  |
| WongKinYiu/yolov7 | models/yolo.py | no-model | Model |  | 8 |  | main class Model (line 508) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| WongKinYiu/yolov7 | models/common.py | parsed | ST2CSPC | 72 | 59 | 60 | vanishing-gradient, deep-no-norm, init-activation-mismatch |  |
| milesial/Pytorch-UNet | unet/unet_model.py | no-model | UNet |  | 0 |  | main class UNet (line 6) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| milesial/Pytorch-UNet | unet/unet_parts.py | parsed | OutConv | 8 | 10 | 4 | clean |  |
| kuangliu/pytorch-cifar | models/resnet.py | parsed | ResNet | 11 | 17 | 9 | clean |  |
| kuangliu/pytorch-cifar | models/vgg.py | parsed | VGG | 6 | 6 | 2 | pool-into-linear-no-flatten | artefact=1 |
| kuangliu/pytorch-cifar | models/densenet.py | parsed | DenseNet | 17 | 9 | 9 | clean |  |
| kuangliu/pytorch-cifar | models/mobilenetv2.py | parsed | MobileNetV2 | 17 | 13 | 9 | pool-into-linear-no-flatten | artefact=1 |
| pytorch/examples | mnist/main.py | parsed | Net | 11 | 6 | 0 | deep-no-norm | not-a-bug=1 |
| pytorch/examples | word_language_model/model.py | parsed | TransformerModel | 5 | 7 | 9 | clean |  |
| jadore801120/attention-is-all-you-need-pytorch | transformer/Models.py | parsed | Transformer | 21 | 7 | 4 | dropout-before-bn |  |
| jadore801120/attention-is-all-you-need-pytorch | transformer/SubLayers.py | parsed | PositionwiseFeedForward | 7 | 10 | 10 | bn-at-output, attention-no-pe | not-a-bug=2 |
| jadore801120/attention-is-all-you-need-pytorch | transformer/Layers.py | parsed | DecoderLayer | 3 | 0 | 0 | attention-no-pe | not-a-bug=1 |
| hyunwoongko/transformer | models/model/transformer.py | no-model | Transformer |  | 0 |  | main class Transformer (line 13) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| hyunwoongko/transformer | models/model/encoder.py | parsed | Encoder | 7 | 0 | 0 | deep-no-norm |  |
| hyunwoongko/transformer | models/blocks/encoder_layer.py | parsed | EncoderLayer | 6 | 2 | 2 | dropout-before-bn x2, bn-at-output, attention-no-pe | not-a-bug=2 |
| hyunwoongko/transformer | models/layers/multi_head_attention.py | parsed | MultiHeadAttention | 5 | 4 | 8 | attention-no-pe, consecutive-linear-no-activation x2 | artefact=2 not-a-bug=1 |
| kyegomez/BitNet | bitnet/bit_transformer.py | parsed | BitNetTransformer | 3 | 4 | 4 | double-norm |  |
| kyegomez/BitNet | bitnet/bitlinear.py | no-model-class |  |  | 0 |  | no class defines both __init__ and forward | no class defines both __init__ and forward (subclass overriding forward only, or pipeline module) |
| Dao-AILab/flash-attention | flash_attn/models/gpt.py | parsed | GPTLMHeadModel | 4 | 3 | 5 | consecutive-linear-no-activation | not-a-bug=1 |
| Dao-AILab/flash-attention | flash_attn/modules/mha.py | parsed | ParallelMHA | 20 | 11 | 19 | deep-no-residual, deep-no-norm | artefact=1 not-a-bug=1 |
| NVIDIA/Megatron-LM | megatron/core/models/gpt/gpt_model.py | parsed | GPTModel | 1 | 0 | 0 | clean |  |
| NVIDIA/Megatron-LM | megatron/core/transformer/transformer_layer.py | no-model | TransformerLayer |  | 0 |  | main class TransformerLayer (line 320) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| NVIDIA/Megatron-LM | megatron/core/transformer/attention.py | no-model | Attention |  | 0 |  | main class Attention (line 288) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| microsoft/LoRA | loralib/layers.py | no-model | ConvLoRA |  | 1 |  | main class ConvLoRA (line 246) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| ashawkey/stable-dreamfusion | nerf/network.py | parsed | NeRFNetwork | 4 | 7 | 5 | clean |  |
| ashawkey/stable-dreamfusion | nerf/renderer.py | no-model | NeRFRenderer |  | 0 |  | main class NeRFRenderer (line 257) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| huggingface/diffusers | src/diffusers/models/unets/unet_2d.py | parsed | UNet2DModel | 8 | 6 | 12 | clean |  |
| huggingface/diffusers | src/diffusers/models/attention.py | parsed | FeedForward | 14 | 26 | 6 | clean |  |
| huggingface/diffusers | src/diffusers/models/unets/unet_2d_condition.py | parsed | UNet2DConditionModel | 4 | 6 | 10 | clean |  |
| open-mmlab/mmdetection | mmdet/models/backbones/resnet.py | parsed | ResNet | 1 | 7 | 0 | clean |  |
| open-mmlab/mmdetection | mmdet/models/backbones/swin.py | parsed | SwinTransformer | 2 | 6 | 1 | clean |  |
| pytorch/audio | src/torchaudio/models/wav2vec2/model.py | no-model | HuBERTPretrainModel |  | 2 |  | main class HuBERTPretrainModel (line 123) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| pytorch/audio | src/torchaudio/models/wav2vec2/components.py | parsed | LogitGenerator | 5 | 20 | 5 | bn-at-output, double-norm x2 | not-a-bug=1 |
| pytorch/audio | src/torchaudio/models/conformer.py | parsed | Conformer | 96 | 19 | 84 | attention-no-pe, double-norm | not-a-bug=1 |
| speechbrain/speechbrain | speechbrain/lobes/models/transformer/Transformer.py | parsed | NormalizedEmbedding | 39 | 6 | 0 | deep-no-norm |  |
| speechbrain/speechbrain | speechbrain/nnet/attention.py | parsed | RoPEMHA | 8 | 24 | 12 | vanishing-gradient, deep-no-norm, linear-after-conv-no-flatten (block), consecutive-linear-no-activation, non-spatial-into-conv, init-activation-mismatch | artefact=1 not-a-bug=1 |
| rwightman/gen-efficientnet-pytorch | geffnet/gen_efficientnet.py | parsed | GenEfficientNet | 3 | 4 | 2 | clean |  |
| rwightman/gen-efficientnet-pytorch | geffnet/mobilenetv3.py | parsed | MobileNetV3 | 2 | 5 | 2 | clean |  |
| xinntao/ESRGAN | RRDBNet_arch.py | parsed | RRDBNet | 18 | 13 | 9 | deep-no-norm |  |
| NVlabs/stylegan2-ada-pytorch | training/networks.py | parsed | Discriminator | 6 | 0 | 0 | linear-after-conv-no-flatten (block), consecutive-linear-no-activation, invalid-output-shape | artefact=2 |
| facebookresearch/fairseq | fairseq/models/transformer/transformer_encoder.py | parsed | TransformerEncoderBase | 4 | 1 | 2 | clean |  |
| facebookresearch/fairseq | fairseq/modules/multihead_attention.py | parsed | MultiheadAttention | 11 | 4 | 20 | deep-no-norm |  |
| facebookresearch/fairseq | fairseq/models/roberta/model.py | parsed | RobertaEncoder | 4 | 6 | 6 | clean |  |
| ultralytics/yolov5 | models/yolo.py | parsed | DetectionModel | 1 | 3 | 2 | clean |  |
| ultralytics/yolov5 | models/common.py | parsed | Classify | 12 | 20 | 16 | bn-at-output | artefact=1 |
| microsoft/Swin-Transformer | models/swin_transformer.py | parsed | SwinTransformer | 1 | 13 | 2 | clean |  |
| google/gemma_pytorch | gemma/model.py | parsed | GemmaForCausalLM | 158 | 0 | 0 | vanishing-gradient |  |
| black-forest-labs/flux | src/flux/model.py | parsed | Flux | 2 | 2 | 4 | clean |  |
| black-forest-labs/flux | src/flux/modules/layers.py | parsed | LastLayer | 22 | 24 | 24 | clean |  |
| mosaicml/llm-foundry | llmfoundry/models/mpt/modeling_mpt.py | parsed | ComposerMPTCausalLM | 4 | 4 | 4 | clean |  |
| eriklindernoren/PyTorch-GAN | implementations/dcgan/dcgan.py | parsed | Discriminator | 12 | 19 | 3 | vanishing-gradient, init-activation-mismatch |  |
| eriklindernoren/PyTorch-GAN | implementations/cyclegan/models.py | parsed | Discriminator | 32 | 26 | 36 | vanishing-gradient, deep-no-norm, init-activation-mismatch | artefact=1 |
| yunjey/pytorch-tutorial | tutorials/02-intermediate/convolutional_neural_network/main.py | parsed | ConvNet | 9 | 10 | 2 | pool-into-linear-no-flatten | artefact=1 |
| yunjey/pytorch-tutorial | tutorials/03-advanced/image_captioning/model.py | parsed | DecoderRNN | 3 | 5 | 7 | clean |  |
| microsoft/unilm | beit/modeling_finetune.py | parsed | VisionTransformer | 1 | 11 | 2 | clean |  |

| Unparsed file cause | Files |
|---|---|
| sub-modules are instances of classes from other files, or are passed in through __init__ | 7 |
| sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) | 6 |
| no class defines both __init__ and forward (subclass overriding forward only, or pipeline module) | 3 |

| Rule | Severity | Measured | Count | Real | Artefact | Not a bug | Unjudged |
|---|---|---|---|---|---|---|---|
| consecutive-linear-no-activation | info | no | 27 | 0 | 5 | 1 | 21 |
| deep-no-norm | info | yes | 18 | 0 | 2 | 3 | 13 |
| vanishing-gradient | info | no | 12 | 0 | 0 | 0 | 12 |
| attention-no-pe | warn | no | 11 | 0 | 3 | 8 | 0 |
| dropout-before-bn | info | no | 9 | 0 | 0 | 0 | 9 |
| double-norm | info | no | 8 | 0 | 0 | 0 | 8 |
| init-activation-mismatch | info | no | 7 | 0 | 0 | 0 | 7 |
| bn-at-output | warn | no | 6 | 0 | 2 | 4 | 0 |
| deep-no-residual | warn | no | 4 | 0 | 4 | 0 | 0 |
| linear-after-conv-no-flatten | block | no | 3 | 0 | 2 | 1 | 0 |
| pool-into-linear-no-flatten | warn | no | 3 | 0 | 3 | 0 | 0 |
| deep-attention-default-init | info | no | 2 | 0 | 0 | 1 | 1 |
| bn-after-activation | warn | no | 2 | 0 | 2 | 0 | 0 |
| flatten-into-attention | warn | no | 2 | 0 | 2 | 0 | 0 |
| invalid-output-shape | warn | yes | 2 | 0 | 2 | 0 | 0 |
| dropout-at-output | warn | no | 1 | 0 | 1 | 0 | 0 |
| moe-no-aux-loss | info | no | 1 | 0 | 0 | 0 | 1 |
| redundant-activation | warn | no | 1 | 0 | 1 | 0 | 0 |
| non-spatial-into-conv | warn | no | 1 | 0 | 1 | 0 | 0 |

Hand-judged blocks: {"total":3,"real":0,"artefact":2,"notABug":1,"unjudged":0}
Hand-judged warns: {"total":34,"real":0,"artefact":22,"notABug":12,"unjudged":0}
Wrote /tmp/discard.json in 357 ms
