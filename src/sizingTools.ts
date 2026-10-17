/**
 * size_training_job: the question to answer before a job takes a GPU
 * reservation. How much memory it needs at this batch, the smallest card that
 * holds it, the largest batch each card holds, a two-card split when one card
 * does not, what one run costs, and the fingerprint to ask `history` what this
 * structure measured the last time it ran.
 *
 * It is a focused wrapper over the vendored plan bundle (the app's
 * `buildPlan`), not a second computation: see lib/sizeJob.ts for the two
 * things it adds on top (removing stored attention scores under SDPA / flash,
 * and answering for the card and batch the caller named). Offline: it never
 * opens a socket unless the caller names an hf: model on a server started with
 * --hf, which is the same rule every other tool follows.
 */
import type { ModelArchitecture } from './lib/types.js';
import type { ToolDef } from './tools.js';
import { sizeJob, type Attention } from './lib/sizeJob.js';
import { loadHFModel } from './sources.js';

export const sizeTrainingJobTool: ToolDef = {
  name: 'size_training_job',
  description:
    'Call this BEFORE submitting or launching any training job (a sbatch, a Kubernetes Job, a cloud GPU reservation, '
    + 'a `python train.py` on a rented card). It answers what the job actually needs: the training peak memory at your '
    + 'batch, split into weights plus optimizer state, activations and stored attention scores; the smallest card that '
    + 'holds it; the largest batch each card (T4, L4, A10G, L40S, A100-40, A100-80, H100-80) holds; a two-card split when '
    + 'one card does not; what one run costs on the requested card or the cheapest that fits; and a one-line verdict such '
    + 'as "fits an A100-40 with 22% headroom" or "will not fit an A100-40: needs ~51 GB, use H100-80 or batch <= 48". '
    + 'It also says whether the graph forward-passes, and returns the structural fingerprint to pass to `history` for '
    + 'what this structure measured the last time it ran. The number is a calibrated estimate of the training peak under '
    + 'AdamW with bf16 autocast, not a measurement: fitted on 141 measured runs across 44 architectures, held-out median '
    + 'error 8% above 1 GB, and the fits / does-not-fit verdict on 16/24/40/80 GB cards agrees with measurement 98% of the '
    + 'time. Recurrent cells are under-counted. The model is the server\'s, or model_path (a .py, a .neurarch.json from '
    + 'neurarch-trace, zoo:<id>, hf:<org/name>), or hf_id. gpu takes "A100-40", "H100", "8xH100" (an N-card request is '
    + 'read both as plain DDP and as sharded FSDP / ZeRO-3). attention defaults to sdpa (the PyTorch and Hugging Face '
    + 'default, which keeps no score matrix); pass eager if the job uses it. Token models are sized at seq_len, or 2048 '
    + 'tokens (the context, if shorter). Runs locally; an hf: model needs the server started with --hf.',
  // hf_id names its own subject, so a hosted server with no model can still answer.
  modelOptional: true,
  inputSchema: {
    type: 'object',
    properties: {
      batch_size: { type: 'integer', minimum: 1, description: 'Global batch size.' },
      gpu: { type: 'string', description: '"A100-40", "H100", "8xH100". Omit for the cheapest that fits.' },
      seq_len: { type: 'integer', minimum: 1, description: 'Tokens per sample (default 2048).' },
      attention: { type: 'string', enum: ['sdpa', 'flash', 'eager'], description: 'Default sdpa.' },
      hf_id: { type: 'string', description: 'Hugging Face org/name to size instead. Needs --hf.' },
      epochs: { type: 'integer', minimum: 1, description: 'To price. Default 10.' },
      samples: { type: 'integer', minimum: 1, description: 'Per epoch, to price (default 50,000).' },
    },
    required: ['batch_size'],
    additionalProperties: false,
  },
  handler: async (
    args: { batch_size: number; gpu?: string; seq_len?: number; attention?: Attention; hf_id?: string; epochs?: number; samples?: number },
    model: ModelArchitecture | undefined,
  ) => {
    if (!Number.isInteger(args.batch_size) || args.batch_size < 1) {
      throw new Error('size_training_job: batch_size must be a positive integer (the job\'s global batch).');
    }
    if (args.attention && !['sdpa', 'flash', 'eager'].includes(args.attention)) {
      throw new Error(`size_training_job: attention "${args.attention}" is not one of sdpa, flash, eager.`);
    }
    let subject = model;
    let publishedParams: number | undefined;
    let source = 'model';
    if (args.hf_id) {
      const id = args.hf_id.replace(/^hf:/i, '').trim();
      const loaded = await loadHFModel(id);
      subject = loaded.model;
      publishedParams = loaded.realParamCount;
      source = `hf:${id} (${loaded.configSource === 'fallback' ? 'a family template, not the real config' : 'config.json'})`;
    }
    if (!subject) {
      throw new Error(
        'size_training_job: name a model. Pass model_path (a path, zoo:<id> or hf:<org/name>), model_source, or hf_id. '
        + 'This server was started without a model of its own.',
      );
    }
    const result = sizeJob(subject, {
      batchSize: args.batch_size,
      gpu: args.gpu,
      seqLen: args.seq_len,
      attention: args.attention,
      epochs: args.epochs,
      samples: args.samples,
    });
    return {
      ...result,
      source,
      ...(publishedParams ? { publishedParams } : {}),
    };
  },
};

export const SIZING_TOOLS: ToolDef[] = [sizeTrainingJobTool];
