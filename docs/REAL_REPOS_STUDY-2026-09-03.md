# The static parser and linter against 59 real PyTorch repositories, re-measured

**Date:** 2026-09-03. Supersedes [REAL_REPOS_STUDY-2026-08-31.md](./REAL_REPOS_STUDY-2026-08-31.md), which stays as the record of the engine this package shipped from 0.13 to 0.15.
**Engine:** `src/vendor/engine.bundle.mjs` rebuilt from neurarch `main` at 60eae74, sha256 `d9f16e085f18000b2b76f0ae5af2ee395b43181df69e3936a8a924c8e09b9a74`, carrying neurarch #128 (same-file classes inlined, root class chosen, source lines kept).
**Command:** `npx tsx scripts/lint-real-repos.mjs --repos-dir <dir> --clone`, same 59 repositories and 116 files as before (`scripts/real-repos.json`).

Two things changed between the runs and both are in the numbers. The parser is the one from #128. And the study now applies the filter `lint_model` applies before an agent sees a finding: a dimension rule (`invalid-output-shape`, `head-dim-divisibility`, `gqa-divisibility`, `linear-in-features-mismatch`) on a layer whose dimension is still source text is held back. That filter landed in the MCP after the first study and was never measured; the count it holds back is reported below rather than hidden.

## Headline numbers, before and after

| Measure | 2026-08-31 | 2026-09-03 |
|---|---|---|
| Files that returned a graph | 73 of 116 (62.9%) | **100 of 116 (86.2%)** |
| Repositories with at least one parsed file | 44 of 59 | 57 of 59 |
| Layers per parsed graph, median | 2 | **8** |
| Parsed graphs with 3 layers or fewer | 48 of 73 | 21 of 100 |
| Parsed graphs with 10 layers or more | 9 of 73 (8% of files) | **47 of 100 (40.5% of files)** |
| Nodes produced vs `nn.*` constructor calls in the parsed files | 450 of 794 | 2946 of 1166 |
| Parsed graphs carrying an unresolved parameter | 60 of 73 | 85 of 100 |
| Dimension findings held back by the MCP filter | not applied | 91 |
| Findings shown (block / warn / info) | 1 / 76 / 26 | 4 / 88 / 190 |
| Parsed graphs with a block | 1 of 73 | 4 of 100 |

The parser now reads the model. Llama, Qwen2, Mistral and Gemma come back as 58-node graphs instead of the one `lm_head` node each returned in August; nanoGPT is 72 nodes instead of 2; Whisper is 135. Nodes exceed constructor calls because a block class instantiated in a loop is inlined once per instance, which is the graph's true size and not a defect. The unparsed 16 files fail for the reasons the first study named and #128 did not address: sub-modules from other files, factory and builder calls, and files with no class holding both `__init__` and `forward`.

## Hand-judged precision

Every block and warn finding was opened against the source and judged as (a) a real issue, (b) a parser artefact, or (c) a true structural fact that is not a bug. Verdicts are in `scripts/real-repos-verdicts.json` with the line numbers they rest on. Info-level findings were not judged this round; the 22 carried over from August are the only ones with a verdict.

| Severity | Total | Real | Artefact | Not a bug |
|---|---|---|---|---|
| block | 4 | **0** | 3 | 1 |
| warn | 88 | **0** | 70 | 18 |
| block + warn | 92 | **0** | 73 | 19 |

Precision on real code is still zero. What changed is the shape of the error. In August 71% of the warnings were one rule judging the parser's own placeholder dimension; the MCP filter now removes those (91 held back here). What remains is one cause wearing several rule names: **the parser records module construction order as data flow and never sees forward().** Every residual add is `x = x + f(x)` in forward, so all 30 `deep-no-residual` warnings fire on networks that have residuals. Every pre-norm transformer chains block i's closing activation into block i+1's opening norm (`bn-after-activation`). Alternative heads selected by config are chained as if consecutive (`redundant-activation`, `flatten-into-attention`, `non-spatial-into-conv`). Three of the four blocks are the same thing: DETR's `input_proj` and `class_embed` are separated by the whole transformer in forward (lines 65-67), SAM's upscaling and hypernetwork MLPs are parallel branches, StyleGAN2 flattens between conv and fc on line 659.

The 19 not-a-bug findings are true statements about a partial view: a file that holds one encoder layer has no positional encoding in it, an encoder's output is normalized hidden states and the rule's text assumes logits, a Linear applied over the channel dim of a 3-D tensor needs no flatten (speechbrain, line 232), and ESRGAN and RWKV-7 do what the rule flags on purpose.

## Real bugs found

Zero, again. No block or warn survived the hand check as a defect in the code it was raised on.

## What this means for the package

- **`trace_model` is the path for real code, and the README already says so.** A graph with forward() in it is what neurarch-trace produces and what this parser structurally cannot. The parser is now good enough to orient (which classes, how many blocks, where the parameters sit) and still not good enough to lint.
- **The next filter is on the rules, not the parser.** Every artefact above comes from an edge the parser inferred from construction order. A graph from source could carry that provenance per edge (`inferred: construction-order`), and the rules that reason about adjacency (`deep-no-residual`, `bn-after-activation`, `redundant-activation`, `dropout-at-output`, the flatten family) would be held back on inferred edges the way dimension rules are held back on unresolved dimensions. That would have removed every artefact in this table and kept every not-a-bug. It is a change to the engine in the main repo, and it is measurable here before it ships.
- Quote this as: the parser reads about 41% of real model files into a recognisable graph (up from 8%), and the linter's findings on such graphs are not yet trustworthy; run `trace_model` before acting on one.

## Per-repo table

`Layers` is the node count of the returned graph. `nn.* calls` is the number of `nn.<Layer>(` constructor calls in the whole file. `Unresolved` is the count of parameters stored as source text. `Verdicts` merges `scripts/real-repos-verdicts.json`.

| Repo | File | Status | Main class | Layers | nn.* calls | Unresolved | Findings | Verdicts |
|---|---|---|---|---|---|---|---|---|
| karpathy/nanoGPT | model.py | parsed | GPT | 72 | 12 | 75 | dropout-before-bn x3, deep-no-residual, consecutive-linear-no-activation | artefact=2 |
| karpathy/minGPT | mingpt/model.py | parsed | GPT | 47 | 14 | 56 | dropout-before-bn x2, deep-no-residual, double-norm | artefact=1 |
| karpathy/nanochat | nanochat/gpt.py | parsed | GPT | 63 | 2 | 2 | deep-no-residual, vanishing-gradient x2, deep-no-norm, consecutive-linear-no-activation x4, init-activation-mismatch | artefact=1 |
| Lightning-AI/litgpt | litgpt/model.py | parsed | RMSNorm | 39 | 15 | 66 | deep-no-residual, deep-no-norm, consecutive-linear-no-activation x4 | artefact=1 |
| huggingface/transformers | src/transformers/models/llama/modeling_llama.py | parsed | LlamaForCausalLM | 58 | 11 | 88 | deep-no-residual, consecutive-linear-no-activation x5 | artefact=1 |
| huggingface/transformers | src/transformers/models/bert/modeling_bert.py | parsed | BertForQuestionAnswering | 23 | 36 | 16 | attention-no-pe, vanishing-gradient, consecutive-linear-no-activation, init-activation-mismatch, deep-attention-default-init | artefact=1 |
| huggingface/transformers | src/transformers/models/gpt2/modeling_gpt2.py | parsed | GPT2ForQuestionAnswering | 44 | 19 | 48 | dropout-before-bn x3, dropout-at-output | artefact=2 |
| huggingface/transformers | src/transformers/models/vit/modeling_vit.py | parsed | ViTForImageClassification | 59 | 16 | 100 | dropout-before-bn, deep-no-residual, consecutive-linear-no-activation x5 | artefact=1 |
| huggingface/transformers | src/transformers/models/whisper/modeling_whisper.py | parsed | WhisperForAudioClassification | 135 | 23 | 234 | deep-no-residual, consecutive-linear-no-activation x4 | artefact=1 |
| huggingface/transformers | src/transformers/models/qwen2/modeling_qwen2.py | parsed | Qwen2ForCausalLM | 58 | 11 | 88 | deep-no-residual, consecutive-linear-no-activation x5 | artefact=1 |
| huggingface/transformers | src/transformers/models/mistral/modeling_mistral.py | parsed | MistralForCausalLM | 58 | 11 | 88 | deep-no-residual, consecutive-linear-no-activation x5 | artefact=1 |
| huggingface/transformers | src/transformers/models/mixtral/modeling_mixtral.py | parsed | MixtralForCausalLM | 46 | 8 | 52 | deep-no-residual, moe-no-aux-loss, consecutive-linear-no-activation x3 | artefact=1 |
| huggingface/transformers | src/transformers/models/gemma/modeling_gemma.py | parsed | GemmaForCausalLM | 58 | 11 | 86 | deep-no-residual, consecutive-linear-no-activation x5 | artefact=1 |
| huggingface/transformers | src/transformers/models/clip/modeling_clip.py | parsed | CLIPForImageClassification | 52 | 22 | 92 | bn-at-output, deep-no-residual, consecutive-linear-no-activation x4, double-norm | artefact=1 not-a-bug=1 |
| huggingface/pytorch-image-models | timm/models/vision_transformer.py | parsed | VisionTransformer | 7 | 14 | 6 | dropout-before-bn, attention-no-pe | artefact=1 |
| huggingface/pytorch-image-models | timm/models/resnet.py | parsed | ResNet | 7 | 14 | 9 | deep-no-norm | artefact=1 |
| huggingface/pytorch-image-models | timm/models/efficientnet.py | no-model | EfficientNetFeatures |  | 1 |  | main class EfficientNetFeatures (line 356) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| huggingface/pytorch-image-models | timm/models/convnext.py | parsed | ConvNeXt | 3 | 3 | 4 | clean |  |
| huggingface/pytorch-image-models | timm/models/swin_transformer.py | parsed | SwinTransformer | 42 | 6 | 12 | deep-no-norm |  |
| pytorch/vision | torchvision/models/resnet.py | parsed | ResNet | 5 | 9 | 3 | clean |  |
| pytorch/vision | torchvision/models/vgg.py | parsed | VGG | 9 | 13 | 1 | deep-no-norm |  |
| pytorch/vision | torchvision/models/densenet.py | parsed | DenseNet | 8 | 16 | 2 | clean |  |
| pytorch/vision | torchvision/models/mobilenetv2.py | parsed | MobileNetV2 | 2 | 3 | 1 | clean |  |
| pytorch/vision | torchvision/models/vision_transformer.py | parsed | VisionTransformer | 1 | 10 | 0 | dropout-at-output | artefact=1 |
| facebookresearch/DiT | models.py | parsed | DiT | 177 | 14 | 2 | deep-no-residual, attention-no-pe, deep-attention-default-init | artefact=2 not-a-bug=1 |
| facebookresearch/dinov2 | dinov2/models/vision_transformer.py | no-model | DinoVisionTransformer |  | 0 |  | main class DinoVisionTransformer (line 45) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| facebookresearch/mae | models_mae.py | parsed | MaskedAutoencoderViT | 35 | 3 | 4 | deep-no-norm | artefact=1 |
| facebookresearch/mae | models_vit.py | no-model-class |  |  | 0 |  | no class defines both __init__ and forward | no class defines both __init__ and forward (subclass overriding forward only, or pipeline module) |
| facebookresearch/detr | models/detr.py | parsed | MLP | 4 | 4 | 6 | linear-after-conv-no-flatten (block), consecutive-linear-no-activation | artefact=1 |
| facebookresearch/detr | models/transformer.py | parsed | TransformerDecoderLayer | 11 | 21 | 15 | attention-no-pe, dropout-at-output, double-norm x2 | artefact=3 not-a-bug=1 |
| facebookresearch/detr | models/backbone.py | no-model | Joiner |  | 0 |  | main class Joiner (line 96) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| facebookresearch/segment-anything | segment_anything/modeling/image_encoder.py | parsed | PatchEmbed | 3 | 5 | 3 | clean |  |
| facebookresearch/segment-anything | segment_anything/modeling/sam.py | no-model | Sam |  | 0 |  | main class Sam (line 18) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| facebookresearch/segment-anything | segment_anything/modeling/mask_decoder.py | parsed | MLP | 25 | 5 | 21 | output-activation, deep-no-residual, vanishing-gradient, deep-no-norm, linear-after-conv-no-flatten (block), redundant-activation | artefact=3 |
| facebookresearch/ConvNeXt | models/convnext.py | parsed | LayerNorm | 1 | 8 | 2 | clean |  |
| facebookresearch/deit | models.py | parsed | DistilledVisionTransformer | 2 | 1 | 4 | consecutive-linear-no-activation | artefact=1 |
| lucidrains/vit-pytorch | vit_pytorch/vit.py | parsed | ViT | 7 | 18 | 8 | dropout-before-bn |  |
| lucidrains/vit-pytorch | vit_pytorch/simple_vit.py | parsed | SimpleViT | 5 | 13 | 7 | double-norm |  |
| lucidrains/x-transformers | x_transformers/x_transformers.py | parsed | XTransformer | 18 | 49 | 16 | clean |  |
| lucidrains/denoising-diffusion-pytorch | denoising_diffusion_pytorch/denoising_diffusion_pytorch.py | parsed | GaussianDiffusion | 12 | 20 | 10 | clean |  |
| lucidrains/PaLM-pytorch | palm_pytorch/palm_pytorch.py | parsed | ParallelTransformerBlock | 7 | 5 | 6 | consecutive-linear-no-activation |  |
| openai/whisper | whisper/model.py | parsed | Whisper | 4 | 2 | 2 | bn-after-activation, bn-at-output | artefact=2 |
| openai/CLIP | clip/model.py | parsed | CLIP | 431 | 32 | 176 | bn-at-output, deep-no-residual, consecutive-linear-no-activation x4, non-spatial-into-conv, double-norm x3, invalid-output-shape | artefact=7 |
| openai/guided-diffusion | guided_diffusion/unet.py | parsed | EncoderUNetModel | 5 | 18 | 0 | attention-no-pe, redundant-activation, flatten-into-attention | artefact=2 not-a-bug=1 |
| state-spaces/mamba | mamba_ssm/modules/mamba_simple.py | parsed | Mamba | 14 | 6 | 9 | deep-no-norm | not-a-bug=1 |
| state-spaces/mamba | mamba_ssm/models/mixer_seq_simple.py | parsed | MambaLMHeadModel | 2 | 2 | 4 | clean |  |
| BlinkDL/RWKV-LM | RWKV-v4neo/src/model.py | parsed | RWKV | 212 | 46 | 302 | dropout-before-bn x2, deep-no-residual, vanishing-gradient, consecutive-linear-no-activation x14, double-norm, init-activation-mismatch | artefact=4 |
| BlinkDL/RWKV-LM | RWKV-v7/rwkv_v7_demo.py | parsed | RWKV | 81 | 15 | 107 | bn-after-activation, deep-no-residual, vanishing-gradient x2, consecutive-linear-no-activation, double-norm, init-activation-mismatch | artefact=1 not-a-bug=1 |
| deepseek-ai/DeepSeek-V3 | inference/model.py | parsed | Transformer | 87 | 0 | 0 | bn-after-activation, deep-no-residual, consecutive-linear-no-activation x5, invalid-output-shape | artefact=3 |
| mistralai/mistral-inference | src/mistral_inference/transformer.py | parsed | Transformer | 1 | 2 | 2 | clean |  |
| meta-llama/llama | llama/model.py | parsed | Transformer | 69 | 0 | 0 | bn-after-activation, deep-no-residual, consecutive-linear-no-activation x4, double-norm, invalid-output-shape | artefact=3 |
| meta-llama/llama3 | llama/model.py | parsed | Transformer | 69 | 0 | 0 | bn-after-activation, deep-no-residual, consecutive-linear-no-activation x4, invalid-output-shape | artefact=3 |
| EleutherAI/gpt-neox | megatron/model/gpt2_model.py | no-model-class |  |  | 0 |  | no class defines both __init__ and forward | no class defines both __init__ and forward (subclass overriding forward only, or pipeline module) |
| EleutherAI/gpt-neox | megatron/model/transformer.py | parsed | NormPipe | 6 | 1 | 1 | invalid-output-shape | artefact=1 |
| jzhang38/TinyLlama | lit_gpt/model.py | parsed | LLaMAMLP | 14 | 9 | 28 | deep-no-residual, deep-no-norm, consecutive-linear-no-activation x2 | artefact=1 |
| CompVis/stable-diffusion | ldm/modules/diffusionmodules/openaimodel.py | parsed | EncoderUNetModel | 5 | 20 | 0 | attention-no-pe, redundant-activation, flatten-into-attention | artefact=2 not-a-bug=1 |
| CompVis/stable-diffusion | ldm/modules/attention.py | parsed | SpatialTransformer | 23 | 22 | 28 | dropout-before-bn, deep-no-residual, consecutive-linear-no-activation x3, double-norm x2 | artefact=1 |
| lllyasviel/ControlNet | cldm/cldm.py | parsed | ControlNet | 2 | 8 | 0 | redundant-activation | artefact=1 |
| ultralytics/ultralytics | ultralytics/nn/tasks.py | no-model | Ensemble |  | 7 |  | main class Ensemble (line 1502) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| ultralytics/ultralytics | ultralytics/nn/modules/block.py | parsed | Proto26 | 8 | 53 | 2 | clean |  |
| WongKinYiu/yolov7 | models/yolo.py | no-model | Model |  | 8 |  | main class Model (line 508) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| WongKinYiu/yolov7 | models/common.py | parsed | ST2CSPC | 72 | 59 | 60 | deep-no-residual, vanishing-gradient, deep-no-norm, redundant-activation, init-activation-mismatch | artefact=2 |
| milesial/Pytorch-UNet | unet/unet_model.py | no-model | UNet |  | 0 |  | main class UNet (line 6) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| milesial/Pytorch-UNet | unet/unet_parts.py | parsed | OutConv | 8 | 10 | 4 | clean |  |
| kuangliu/pytorch-cifar | models/resnet.py | parsed | ResNet | 11 | 17 | 9 | clean |  |
| kuangliu/pytorch-cifar | models/vgg.py | parsed | VGG | 6 | 6 | 2 | pool-into-linear-no-flatten | artefact=1 |
| kuangliu/pytorch-cifar | models/densenet.py | parsed | DenseNet | 17 | 9 | 9 | clean |  |
| kuangliu/pytorch-cifar | models/mobilenetv2.py | parsed | MobileNetV2 | 17 | 13 | 9 | pool-into-linear-no-flatten | artefact=1 |
| pytorch/examples | mnist/main.py | parsed | Net | 11 | 6 | 0 | deep-no-norm | not-a-bug=1 |
| pytorch/examples | word_language_model/model.py | parsed | TransformerModel | 5 | 7 | 9 | clean |  |
| jadore801120/attention-is-all-you-need-pytorch | transformer/Models.py | parsed | Transformer | 21 | 7 | 4 | dropout-before-bn, duplicate-positional-encoding |  |
| jadore801120/attention-is-all-you-need-pytorch | transformer/SubLayers.py | parsed | PositionwiseFeedForward | 7 | 10 | 10 | dropout-before-bn, bn-at-output, attention-no-pe, consecutive-linear-no-activation x2 | not-a-bug=3 |
| jadore801120/attention-is-all-you-need-pytorch | transformer/Layers.py | parsed | DecoderLayer | 3 | 0 | 0 | attention-no-pe | not-a-bug=1 |
| hyunwoongko/transformer | models/model/transformer.py | no-model | Transformer |  | 0 |  | main class Transformer (line 13) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| hyunwoongko/transformer | models/model/encoder.py | parsed | Encoder | 7 | 0 | 0 | deep-no-norm |  |
| hyunwoongko/transformer | models/blocks/encoder_layer.py | parsed | EncoderLayer | 6 | 2 | 2 | dropout-before-bn x2, bn-at-output, attention-no-pe | not-a-bug=2 |
| hyunwoongko/transformer | models/layers/multi_head_attention.py | parsed | MultiHeadAttention | 5 | 4 | 8 | attention-no-pe, consecutive-linear-no-activation x2 | artefact=2 not-a-bug=1 |
| kyegomez/BitNet | bitnet/bit_transformer.py | parsed | BitNetTransformer | 3 | 4 | 4 | bn-at-output, double-norm | not-a-bug=1 |
| kyegomez/BitNet | bitnet/bitlinear.py | no-model-class |  |  | 0 |  | no class defines both __init__ and forward | no class defines both __init__ and forward (subclass overriding forward only, or pipeline module) |
| Dao-AILab/flash-attention | flash_attn/models/gpt.py | parsed | GPTLMHeadModel | 4 | 3 | 5 | consecutive-linear-no-activation | not-a-bug=1 |
| Dao-AILab/flash-attention | flash_attn/modules/mha.py | parsed | ParallelMHA | 20 | 11 | 19 | deep-no-residual, deep-no-norm, consecutive-linear-no-activation x2, duplicate-positional-encoding | artefact=2 not-a-bug=1 |
| NVIDIA/Megatron-LM | megatron/core/models/gpt/gpt_model.py | parsed | GPTModel | 1 | 0 | 0 | clean |  |
| NVIDIA/Megatron-LM | megatron/core/transformer/transformer_layer.py | no-model | TransformerLayer |  | 0 |  | main class TransformerLayer (line 318) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| NVIDIA/Megatron-LM | megatron/core/transformer/attention.py | no-model | Attention |  | 0 |  | main class Attention (line 288) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| microsoft/LoRA | loralib/layers.py | no-model | ConvLoRA |  | 1 |  | main class ConvLoRA (line 246) yielded zero layers | sub-modules built by factory or builder calls (create_conv2d, builder(), get_down_block, build_module, parse_model(yaml), build_conv_layer, norm_class(), conv_module(), embed_layer(), block_fn) |
| ashawkey/stable-dreamfusion | nerf/network.py | parsed | NeRFNetwork | 4 | 7 | 5 | clean |  |
| ashawkey/stable-dreamfusion | nerf/renderer.py | no-model | NeRFRenderer |  | 0 |  | main class NeRFRenderer (line 257) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| huggingface/diffusers | src/diffusers/models/unets/unet_2d.py | parsed | UNet2DModel | 8 | 6 | 12 | clean |  |
| huggingface/diffusers | src/diffusers/models/attention.py | parsed | FeedForward | 14 | 26 | 6 | double-norm, duplicate-positional-encoding |  |
| huggingface/diffusers | src/diffusers/models/unets/unet_2d_condition.py | parsed | UNet2DConditionModel | 4 | 6 | 10 | clean |  |
| open-mmlab/mmdetection | mmdet/models/backbones/resnet.py | parsed | ResNet | 1 | 7 | 0 | clean |  |
| open-mmlab/mmdetection | mmdet/models/backbones/swin.py | parsed | SwinTransformer | 2 | 6 | 1 | dropout-at-output | artefact=1 |
| pytorch/audio | src/torchaudio/models/wav2vec2/model.py | no-model | HuBERTPretrainModel |  | 2 |  | main class HuBERTPretrainModel (line 123) yielded zero layers | sub-modules are instances of classes from other files, or are passed in through __init__ |
| pytorch/audio | src/torchaudio/models/wav2vec2/components.py | parsed | LogitGenerator | 5 | 20 | 5 | dropout-before-bn, bn-at-output, double-norm x2 | not-a-bug=1 |
| pytorch/audio | src/torchaudio/models/conformer.py | parsed | Conformer | 96 | 19 | 84 | dropout-before-bn x2, bn-at-output, deep-no-residual, attention-no-pe, double-norm | artefact=1 not-a-bug=2 |
| speechbrain/speechbrain | speechbrain/lobes/models/transformer/Transformer.py | parsed | NormalizedEmbedding | 39 | 6 | 0 | deep-no-norm, dropout-at-output, duplicate-positional-encoding | artefact=1 |
| speechbrain/speechbrain | speechbrain/nnet/attention.py | parsed | RoPEMHA | 8 | 24 | 12 | vanishing-gradient, deep-no-norm, linear-after-conv-no-flatten (block), redundant-activation, consecutive-linear-no-activation x2, non-spatial-into-conv, init-activation-mismatch | artefact=2 not-a-bug=1 |
| rwightman/gen-efficientnet-pytorch | geffnet/gen_efficientnet.py | parsed | GenEfficientNet | 3 | 4 | 2 | clean |  |
| rwightman/gen-efficientnet-pytorch | geffnet/mobilenetv3.py | parsed | MobileNetV3 | 2 | 5 | 2 | clean |  |
| xinntao/ESRGAN | RRDBNet_arch.py | parsed | RRDBNet | 18 | 13 | 9 | deep-no-residual, deep-no-norm | artefact=1 not-a-bug=1 |
| NVlabs/stylegan2-ada-pytorch | training/networks.py | parsed | Discriminator | 6 | 0 | 0 | linear-after-conv-no-flatten (block), consecutive-linear-no-activation, invalid-output-shape x2 | artefact=3 |
| facebookresearch/fairseq | fairseq/models/transformer/transformer_encoder.py | parsed | TransformerEncoderBase | 4 | 1 | 2 | bn-at-output | not-a-bug=1 |
| facebookresearch/fairseq | fairseq/modules/multihead_attention.py | parsed | MultiheadAttention | 11 | 4 | 20 | deep-no-residual, deep-no-norm, consecutive-linear-no-activation x3 | artefact=1 |
| facebookresearch/fairseq | fairseq/models/roberta/model.py | parsed | RobertaEncoder | 4 | 6 | 6 | clean |  |
| ultralytics/yolov5 | models/yolo.py | parsed | DetectionModel | 1 | 3 | 2 | clean |  |
| ultralytics/yolov5 | models/common.py | parsed | Classify | 12 | 20 | 16 | bn-at-output | artefact=1 |
| microsoft/Swin-Transformer | models/swin_transformer.py | parsed | SwinTransformer | 1 | 13 | 2 | clean |  |
| google/gemma_pytorch | gemma/model.py | parsed | GemmaForCausalLM | 158 | 0 | 0 | bn-at-output, deep-no-residual, vanishing-gradient, redundant-activation, consecutive-linear-no-activation, double-norm x3, invalid-output-shape | artefact=3 not-a-bug=1 |
| black-forest-labs/flux | src/flux/model.py | parsed | Flux | 2 | 2 | 4 | consecutive-linear-no-activation | artefact=1 |
| black-forest-labs/flux | src/flux/modules/layers.py | parsed | LastLayer | 22 | 24 | 24 | deep-no-residual, consecutive-linear-no-activation x2, double-norm x2 | artefact=1 |
| mosaicml/llm-foundry | llmfoundry/models/mpt/modeling_mpt.py | parsed | ComposerMPTCausalLM | 4 | 4 | 4 | dropout-at-output | artefact=1 |
| eriklindernoren/PyTorch-GAN | implementations/dcgan/dcgan.py | parsed | Discriminator | 12 | 19 | 3 | vanishing-gradient, init-activation-mismatch |  |
| eriklindernoren/PyTorch-GAN | implementations/cyclegan/models.py | parsed | Discriminator | 32 | 26 | 36 | deep-no-residual, vanishing-gradient, deep-no-norm, init-activation-mismatch | artefact=2 |
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
| consecutive-linear-no-activation | info | no | 103 | 0 | 11 | 1 | 91 |
| deep-no-residual | warn | no | 30 | 0 | 30 | 0 | 0 |
| double-norm | info | no | 23 | 0 | 2 | 0 | 21 |
| dropout-before-bn | info | no | 21 | 0 | 1 | 1 | 19 |
| deep-no-norm | info | yes | 18 | 0 | 3 | 4 | 11 |
| vanishing-gradient | info | no | 12 | 0 | 0 | 0 | 12 |
| attention-no-pe | warn | no | 11 | 0 | 3 | 8 | 0 |
| bn-at-output | warn | no | 11 | 0 | 4 | 7 | 0 |
| init-activation-mismatch | info | no | 8 | 0 | 0 | 0 | 8 |
| invalid-output-shape | warn | yes | 8 | 0 | 8 | 0 | 0 |
| redundant-activation | warn | no | 7 | 0 | 6 | 1 | 0 |
| dropout-at-output | warn | no | 6 | 0 | 6 | 0 | 0 |
| bn-after-activation | warn | no | 5 | 0 | 4 | 1 | 0 |
| linear-after-conv-no-flatten | block | no | 4 | 0 | 3 | 1 | 0 |
| duplicate-positional-encoding | info | no | 4 | 0 | 1 | 0 | 3 |
| pool-into-linear-no-flatten | warn | no | 3 | 0 | 3 | 0 | 0 |
| deep-attention-default-init | info | no | 2 | 0 | 0 | 1 | 1 |
| non-spatial-into-conv | warn | no | 2 | 0 | 2 | 0 | 0 |
| flatten-into-attention | warn | no | 2 | 0 | 2 | 0 | 0 |
| moe-no-aux-loss | info | no | 1 | 0 | 0 | 0 | 1 |
| output-activation | info | no | 1 | 0 | 0 | 0 | 1 |

Hand-judged blocks: {"total":4,"real":0,"artefact":3,"notABug":1,"unjudged":0}
Hand-judged warns: {"total":88,"real":0,"artefact":70,"notABug":18,"unjudged":0}
Wrote /Users/xingao/Projects/neurarch-mcp/docs/real-repos-results.json in 259 ms
