# Transactional Browser-Agent Training Design

Date: 2026-07-24  
Status: approved direction; implementation plan pending

## 1. Goal

Train a separate LoRA on the frozen Gemma-4-31B base so the model can complete
long-horizon, multi-turn browser tasks for:

- restaurant reservations;
- restaurant delivery or pickup;
- café and coffee-shop orders;
- bakery orders;
- similar local hospitality and prepared-food transactions.

The target capability is the complete interaction loop:

1. understand the user's request;
2. search for appropriate venues or items;
3. inspect availability, menus, prices, fees, and terms;
4. ask only the necessary preference or clarification questions;
5. select the exact slot or item;
6. verify the reservation summary or basket;
7. fill forms using secure credential indirection;
8. decide whether to commit automatically or request approval;
9. submit when authorized;
10. verify and report the final confirmation.

This is a model-training lane, not a plan to build an entire consumer booking
platform. Side scripts exist only to collect and normalize traces, expose
structured browser state, simulate transactional sites, protect credentials,
execute actions, and score outcomes.

## 2. Scope and non-goals

### In scope

- Text-first browser control using accessibility-tree or bounded DOM state.
- Search, navigation, clicking, selection, form filling, and submission.
- Deep multi-step conversations where constraints are supplied or changed over
  several user turns.
- Restaurant booking, prepared-food ordering, pickup, and delivery.
- Hybrid autonomy: safe exact matches may be committed automatically; ambiguous
  or financially risky actions require user confirmation.
- Saved-payment use inside a user-configured authorization envelope.
- Robust recovery from unavailable slots, unavailable items, stale pages,
  validation failures, session expiry, and navigation errors.
- A separate browser LoRA whose effect can be evaluated and rolled back without
  changing the coding policy.

### Out of scope

- General retail, supermarkets, groceries, large stores, travel, banking,
  subscriptions, ticket resale, medical transactions, or high-value purchases.
- Training directly on raw card numbers, CVVs, passwords, session cookies, or
  other secrets.
- Letting a side script perform the reasoning or choose the venue/item on the
  model's behalf.
- Unbounded live-web reinforcement learning before simulator safety gates pass.
- Merging the browser LoRA into the coding LoRA before an explicit no-regression
  evaluation.

## 3. Core behavior contract

The model learns one repeated control loop:

```text
understand goal
  -> inspect browser state
  -> update constraints and transaction state
  -> decide ask versus act
  -> execute one bounded action
  -> verify the resulting state
  -> recover, continue, commit, or finish
```

Each model turn receives:

- the system policy;
- the user's conversation;
- the standing authorization envelope;
- a compact transaction-state summary;
- a bounded structured browser observation;
- recent action results and errors.

Each model turn produces exactly one of:

```text
browser.search
browser.open
browser.click
browser.fill
browser.select
browser.back
ask_user
inspect_transaction
commit_transaction
finish
```

The model, not a workflow engine, selects the action. The runtime validates the
schema and applies hard safety limits but does not decide what the user wants.

### 3.1 Transaction-state representation

The model is trained to maintain a compact state object:

```json
{
  "domain": "reservation|pickup|delivery",
  "goal": "one normalized sentence",
  "constraints": {
    "venue": {"value": "string|null", "source": "user|inferred|null"},
    "date": {"value": "ISO date|null", "source": "user|inferred|null"},
    "time": {"value": "local time|null", "source": "user|inferred|null"},
    "party_size": {"value": "integer|null", "source": "user|inferred|null"},
    "items": [{"name": "string", "quantity": 1, "modifiers": []}],
    "dietary": [],
    "fulfillment": "reservation|pickup|delivery|null",
    "location": "string|null",
    "budget": {"currency": "USD", "maximum": "number|null"}
  },
  "unresolved": [],
  "selected_candidate": null,
  "observed_total": null,
  "unexpected_terms": [],
  "authorization": {
    "saved_payment_allowed": true,
    "per_transaction_cap": 40.0,
    "auto_commit_allowed": true
  },
  "status": "searching|clarifying|configuring|reviewing|committed|failed"
}
```

The state is supervised as structured output during early SFT. Later preference
and RL stages reward correct behavior rather than exact wording, so the model
does not overfit one serialization.

### 3.2 Hybrid commit policy

Automatic commitment is correct only when all conditions hold:

- every material user constraint has an exact observed match;
- the requested date/time or item is available without substitution;
- the basket contains exactly the requested items and quantities;
- fulfillment method and location match;
- the total is within the user's standing cap;
- a saved payment method is already authorized when money is charged;
- no new fee, deposit, tip, subscription, cancellation penalty, or material
  term appears;
- the page exposes a clear final summary;
- the model has verified that the action will not create a duplicate
  reservation or order.

The model must ask the user before commitment when any of these occurs:

- an alternative time, venue, item, size, flavor, modifier, or quantity;
- an unavailable requested option;
- missing party size, date, time, fulfillment method, address, or required
  dietary choice;
- a deposit, prepayment, cancellation penalty, mandatory tip, or unexpected fee;
- a total above the standing cap;
- a new payment method or missing secure credential;
- multiple materially different candidates with no stated preference;
- uncertain basket contents or uncertain final-action semantics;
- a possible duplicate order or reservation.

The model must not ask redundant questions when the answer is already explicit
in the conversation or verified browser state.

### 3.3 Credential boundary

Raw payment and authentication secrets never enter:

- model prompts;
- model outputs;
- training traces;
- reward logs;
- evaluation artifacts.

The model sees only non-secret metadata such as:

```json
{
  "credential_alias": "personal_visa",
  "network": "visa",
  "last_four": "1234",
  "billing_country_match": true,
  "available": true
}
```

For training and evaluation, a deterministic secure-broker stub fills synthetic
credentials. A real runtime would expose the same interface to a secret manager.

## 4. Browser observation and action format

The first lane is structured-browser-first:

- accessibility tree;
- selected DOM attributes;
- visible text;
- URL and title;
- active tab;
- relevant form state;
- concise action history;
- optional screenshot reference as fallback evidence.

The observation builder should retrieve the most relevant interactive elements
instead of sending full HTML. This follows the evidence from
[Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web) and
[WebLINX](https://github.com/mcgill-nlp/weblinx) that real pages need
retrieval/pruning for tractable model context.

Every element receives a stable observation-scoped identifier:

```json
{
  "element_id": "e17",
  "role": "button",
  "name": "Reserve 7:30 PM",
  "value": null,
  "disabled": false,
  "visible": true
}
```

Actions reference only identifiers present in the current observation. A stale
identifier returns an observation error and trains recovery rather than silent
failure.

The recommended environment interface is compatible with
[BrowserGym](https://github.com/ServiceNow/BrowserGym), which already supports
open-ended interactive tasks and conversational user turns.

### 4.1 Canonical actions across multiple browser surfaces

The same hidden transaction task should be renderable through several
observation surfaces:

- the native accessibility-tree interface;
- a bounded DOM representation;
- a BrowserGym-compatible representation;
- a recovery variant that returns stale-element, validation, or navigation
  errors.

All surfaces map to one canonical semantic action IR. A task that means "select
the 7:30 PM slot" must have the same effect regardless of how the page labels or
orders its elements. Observation variation is desirable; incompatible action
semantics are not.

This adapts Poolside's multi-harness training without copying its known harness
overfitting failure. The model should learn to bind the current tool definition
and browser state to stable transaction semantics, not memorize one site's
element order or one harness's incidental syntax.

### 4.2 Train/deploy rendering invariant

The production chat template, reasoning parser, tool parser, observation
renderer, and action schema must be the same implementations used to generate
and replay training rollouts. After every generated action, the collector must
assert:

```text
render(conversation_history_with_raw_assistant_tokens)
  == decoded_rollout_prefix
```

Whitespace, reasoning-channel boundaries, tool-call fields, and observation
pairing are part of this invariant. A mismatch is a blocking format failure, not
something to tolerate during training.

When thinking is enabled, prior reasoning blocks remain in the conversation
history. Raw credentials remain excluded from both reasoning and visible
history. Training includes concise no-reasoning trajectories as well as
reasoning-preserved trajectories so low-risk orders do not acquire unnecessary
deliberation.

### 4.3 Risk-adaptive reasoning budget

Long reasoning is a capability, not a default objective. Laguna reports large
agentic gains from thinking mode, but also much higher completion-token use and
occasional overthinking. Browser transactions need effort proportional to
ambiguity and risk:

- exact, low-risk, no-payment tasks should use a concise state check and act;
- preference matching, alternatives, or recoverable errors may use a moderate
  reasoning budget;
- payment, deposits, conflicting constraints, possible duplicates, or deceptive
  page content may use a larger budget before asking or committing.

Training manifests label these three effort bands. Evaluation reports reasoning
tokens, browser actions, and user turns per successful transaction so a more
verbose checkpoint cannot appear better merely by consuming unbounded
test-time compute.

## 5. Training architecture

Use a separate adapter:

```text
frozen Gemma-4-31B base
  + browser_transaction_lora_r32
```

Initial configuration:

- LoRA rank: 32;
- LoRA alpha: 32 or the existing repository default for r32;
- base weights: frozen;
- context: 32,768 initially, with a 49,152-token compatibility gate;
- assistant-turn-only loss;
- native Gemma tool format;
- no coding-adapter merge during the first lane.

The browser adapter is loaded only for browser tasks. Adapter composition with
the coding policy is a later experiment requiring coding and browser
no-regression gates.

### 5.1 Research-derived constraints

The plan incorporates two complementary published approaches:

- Poolside's
  [Laguna M.1/XS.2 technical report](https://poolside.ai/assets/laguna/laguna-m1-xs2-technical-report.pdf)
  and
  [Laguna S 2.1 release report](https://poolside.ai/blog/introducing-laguna-s-2-1)
  motivate verified environment generation, mixed reasoning and direct-action
  supervision, multi-harness rollouts, exact train/deploy template alignment,
  difficulty-bucketed online RL, and turn-local tool-error penalties.
- DeepReinforce's
  [Ornith-1.0 description](https://deep-reinforce.com/ornith_1_0.html)
  motivates an optional self-scaffolding phase in which the policy learns an
  inner memory, retrieval, recovery, and verification strategy while the outer
  safety boundary remains immutable.

These are behavioral and systems lessons, not scale-equivalent recipes. Laguna
S 2.1 is a 118B-total-parameter MoE trained from scratch using a corpus and
compute budget many orders of magnitude larger than a Gemma-4-31B LoRA. Ornith
does not publish enough training code or ablations to treat its reported gains
as independently reproduced. Neither source justifies expecting comparable
absolute benchmark gains from this lane.

The immediately transferable requirements are:

1. every positive task must have a deterministic verifier;
2. task admission must prove that the verifier discriminates correct from
   incorrect behavior;
3. training and deployment must share the exact interaction contract;
4. RL should use tasks on which the current policy sometimes, but not always,
   succeeds;
5. the model may improve its inner work strategy but never the authorization,
   credential, tool, or verifier boundary;
6. rollout scale and environment diversity matter more than repeatedly training
   on a few hundred stylistically attractive traces.

## 6. Five-stage training curriculum

### Stage 0: Verified environment factory

Purpose:

- create browser tasks whose outcome can be scored without an LLM judge;
- establish site, layout, policy, and user-interaction diversity before
  trajectory collection;
- prevent false-positive environments from poisoning both SFT and RL.

Each generated environment contains:

- an initial browser/database state;
- a user request and hidden normalized constraints;
- an authorization envelope;
- hidden inventory, schedule, pricing, payment, and policy state;
- a gold final-state predicate;
- counterfactual failure predicates;
- a reproducible seed and environment revision.

Admission uses a two-sided check:

1. the gold action sequence must reach the exact verified target state;
2. a no-op and at least two plausible wrong continuations must fail.

Wrong continuations include a near-match time or item, an extra basket item, an
unapproved fee, an unauthorized commit, and a duplicate submission where
applicable. Environments whose verifier accepts a no-op or plausible wrong state
are rejected.

Target scale:

- pilot: 2,000-4,000 admitted environments;
- promoted corpus: 10,000-20,000 unique environments;
- no single page template above 10%;
- at least 50 distinct merchant/layout families, with procedural variation
  inside each family.

### Stage 1: Verified multi-turn SFT

Purpose:

- teach the browser action grammar;
- teach compact transaction-state tracking;
- teach early grounding and decisive progress;
- teach clarification at genuine ambiguity;
- teach verification after every material mutation;
- teach correct final summaries and confirmation handling.

Sources:

1. Revision-pinned WebLINX 1.1 training demonstrations for conversational
   navigation structure.
2. Revision-pinned Mind2Web training data for element grounding and action
   selection. Benchmark test splits never enter training.
3. Synthetic restaurant/café/bakery environments generated from explicit task
   templates.
4. Execution-verified teacher rollouts from stronger models, rendered into the
   exact Gemma/BrowserGym action convention.
5. Model self-rollouts only after a deterministic verifier proves the outcome.

SFT rows must be execution-verified or derive from a deterministic simulator
gold trajectory. Do not trust publisher success labels alone.

Each admitted environment can produce several verified variants:

- a concise action-first trajectory without visible reasoning;
- a reasoning-preserved trajectory;
- one or more constraint-augmented variants;
- a recovery trajectory when the perturbation is recoverable;
- alternate browser-surface renderings with the same semantic actions.

Target pilot dataset:

- 5,000-10,000 trajectories;
- collected from at least 2,000 unique environments;
- used to prove action grammar, learning direction, and safety before the larger
  collection.

Target promoted dataset:

- 30,000-60,000 verified trajectory variants;
- collected from 10,000-20,000 unique environments;
- approximate token mixture:
  - 40% reasoning-preserved verified trajectories;
  - 30% concise action-first verified trajectories;
  - 20% constraint-augmented or recovery variants;
  - 10% general multi-turn and instruction-retention samples;
- at least 40% multi-turn user interactions;
- at least 35% cases requiring one or more clarifications;
- 30-40% safe auto-commit cases;
- 30-40% confirmation-required cases;
- at least 25% recovery trajectories;
- balanced reservation, pickup, and delivery domains;
- no single site template above 10%.

Trace shaping:

- preserve all user turns and post-action verification;
- remove redundant pre-decision browsing;
- retain the last relevant observations before each material decision;
- never remove the observation supporting a chosen slot/item;
- never supervise hidden credentials;
- cap each observation at 2,400-4,000 characters;
- reject loops and identical consecutive actions;
- require the first relevant browser action by action 3;
- require the first material selection or clarification by action 8.

Expected pilot training time on GPU1:

- one-step smoke: 10-25 minutes;
- 5-step behavior smoke: 45-120 minutes;
- one pilot epoch: 12-30 GPU hours, depending on final row lengths.

Expected promoted-corpus training time:

- one epoch: approximately 2-6 GPU days on a single GPU1;
- use checkpoint evaluation and early stopping rather than committing to three
  epochs in advance;
- shorten observations and pack compatible rows before reducing environment
  diversity.

Expected gain:

- approximately +15 to +25 percentage points in structured action completion
  over the raw-base baseline;
- substantial tool-format improvement;
- only modest ask-versus-act calibration until Stage 2.

These are planning estimates, not promotion claims. The raw-base benchmark must
be measured first.

### Stage 2: Decision-point preference optimization

Purpose:

- teach when to ask versus act;
- discourage redundant confirmation;
- prevent unsafe automatic commitment;
- prefer exact matches over plausible substitutes;
- prefer basket verification over premature submission.

Build preference pairs at the same browser state:

```text
chosen: ask about the unexpected $10 deposit
rejected: submit the reservation automatically
```

```text
chosen: commit the exact 7:30 PM reservation with no payment requirement
rejected: ask the user to confirm the already explicit time
```

```text
chosen: remove the extra pastry and re-inspect the basket
rejected: place the order with the extra item
```

Pair sources:

- branches from simulator states;
- teacher rankings;
- verifier-labeled model failures;
- counterfactual changes to availability, fees, basket contents, or policy;
- adversarial page content attempting to override user intent.

Target dataset:

- 15,000-30,000 preference pairs;
- equal representation of over-asking and under-asking errors;
- at least 25% irreversible-action decisions;
- at least 20% unexpected-fee or changed-availability decisions;
- at least 15% prompt-injection or deceptive-interface negatives.

Training method:

- begin with DPO or KTO on the separate browser LoRA;
- compare against an SFT-only checkpoint on an identical held-out decision set;
- do not accept a checkpoint based on training loss alone.

Expected training time:

- 4-8 GPU hours plus 2-4 hours of decision-point evaluation.

Expected gain:

- +10 to +20 percentage points in ask-versus-act accuracy;
- 30-50% relative reduction in redundant questions;
- materially lower unsafe auto-commit rate.

### Stage 3: Multi-turn RLVR

Purpose:

- optimize end-to-end success rather than imitation;
- learn recovery from changing state;
- reduce loops and unnecessary steps;
- calibrate autonomy using actual transaction outcomes.

Use reproducible local environments first. Each episode randomizes:

- venue, menu, inventory, and opening hours;
- party size and time availability;
- prices, taxes, delivery fees, mandatory tips, and deposits;
- cancellation rules;
- user preferences and standing authorization;
- late item/slot unavailability;
- validation errors and expired sessions;
- deceptive page text and prompt injection;
- user responses to clarification questions.

The user simulator should follow the turn-based policy/tool pattern used by
[τ²-bench](https://github.com/sierra-research/tau2-bench), adapted to
hospitality transactions.

Initial RL configuration:

- measure four baseline rollouts per task before RL admission;
- always-solved tasks become regression tests;
- never-solved tasks return to teacher collection or curriculum SFT;
- admit tasks with a measured pass rate between 25% and 75%;
- sample admitted tasks toward the harder end while retaining successful
  trajectories in every group;
- 2,000-5,000 admitted simulator tasks;
- 4 generations per prompt initially;
- increase to 6 only after the memory smoke;
- 100-step smoke, then at most 300 steps per round;
- checkpoint every 25 steps;
- temperature 0.9-1.0 for exploration;
- group-preserving gradient accumulation;
- no real charges or bookings.

Rollouts use the exact production browser action API, chat template, reasoning
history, recovery behavior, and orchestration layer. Changing the deployed
harness creates a new training/evaluation condition and requires a same-policy
control run.

Start with the existing group-relative RL implementation. Laguna reports better
stability with a CISPO-style clipped REINFORCE objective than with GRPO/GSPO,
but an algorithm change cannot repair sparse rewards. Consider a controlled
CISPO comparison only after the 100-step run demonstrates:

- at least 25% exact-success trajectories;
- both success and failure inside most sampled groups;
- nonzero high-tier reward at every checkpoint interval;
- no sustained policy-format drift.

Only after simulator promotion should a narrow online-RL/expert-iteration lane
be considered. [OpenWebRL](https://openwebrl.github.io/) provides evidence that
online multi-turn RL can produce strong web-agent gains from a relatively small
initialization set, but the live-web phase remains a later, separately approved
experiment.

Expected time:

- simulator episode generation: 1-3 days CPU/browser time;
- 100-step smoke: 3-6 GPU hours;
- 300-step round: 8-16 GPU hours;
- behavioral evaluation: 6-12 hours.

Expected gain:

- +5 to +15 percentage points in end-to-end success over the best SFT/DPO
  checkpoint;
- fewer repeated actions and better recovery;
- uncertain safety gain unless the negative reward and canary gates are strict.

### Stage 4: Optional self-scaffolded RLVR

This phase adapts Ornith's self-scaffolding idea only after ordinary simulator
RLVR produces a dense, stable execution signal.

Before acting, the policy emits a bounded internal scaffold:

```json
{
  "state_fields": ["requested_time", "party_size", "observed_total"],
  "retrieval_priorities": ["availability", "fees", "final_summary"],
  "ask_if": ["material constraint missing", "near-match only"],
  "commit_if": ["all exact", "inside authorization", "no new terms"],
  "recovery_order": ["refresh observation", "reacquire element", "ask user"],
  "verify_before_finish": ["confirmation id", "exact basket or booking"]
}
```

Allowed scaffold content is limited to:

- memory organization;
- browser-state retrieval priorities;
- ask-versus-act checks;
- error recovery order;
- verification strategy.

The scaffold cannot alter:

- the browser action surface;
- credential access;
- authorization or spending limits;
- commit policy;
- hidden environment state;
- verifier logic;
- reward computation.

For each task, sample four candidate scaffolds and one rollout per scaffold.
Apply the deterministic rollout reward to both the scaffold and its resulting
actions, retain successful category-level strategies, and contrast them against
unsafe or inefficient scaffolds. Do not optimize free-form scaffolds without a
schema and forbidden-key monitor.

Stage-4 smoke:

- 100 held-out tasks;
- four scaffold candidates per task;
- synchronous rollouts initially;
- compare against the same Stage-3 policy with a fixed hand-written scaffold.

Promotion requires:

- at least +5 percentage points exact transaction success, or at least 20%
  relative recovery improvement;
- zero additional unsafe commitments, credential events, or duplicates;
- no more than 15% increase in median browser actions or user turns;
- stable scaffold-schema compliance above 99%.

Asynchronous rollout/training is deferred. If later needed, every trajectory
must record the behavior-policy version, stale trajectories must be downweighted,
and trajectories older than a fixed optimizer-step limit must be dropped.

## 7. Reward design

The only positive terminal RL reward is exact verifier success. Necessary
clarification, safe auto-commit, recovery, and final verification remain
separate evaluation metrics and preference-training labels; making them large
independent positive RL rewards would let the model collect reward without
finishing the transaction.

| Component | RL treatment |
|---|---:|
| Exact verified final transaction state | +1.00 terminal |
| Ordinary task failure | 0.00 terminal |
| Timeout or maximum steps | 0.00 terminal |
| Malformed action or chat-template violation | -0.10 on offending turn |
| Tool execution, stale-element, or invalid-selector error | -0.05 on offending turn |
| Identical consecutive action | -0.10 on offending turn |
| Premature finish without final-state evidence | -0.10 on final turn |
| Wrong item, slot, party size, fulfillment, fee, or basket | terminate; 0.00 terminal |
| Missed required confirmation | abort; 0.00 and exclude RL update |
| Unauthorized or above-cap commitment | abort; 0.00 and exclude RL update |
| Credential leakage | abort; 0.00 and exclude RL update |
| Duplicate order/reservation | abort; 0.00 and exclude RL update |

Hard safety failures are blocked by the deterministic environment monitor,
excluded from the RL policy-gradient update, retained as preference negatives,
and reported individually. This prevents an unsafe trajectory from influencing
the learned scaffold while still teaching the distinction through Stage-2
preference data. A frozen LLM judge may veto a technically valid but
intent-violating trajectory; it never supplies the primary positive reward.

Do not add a minimum-action-count reward. For transactions, extra actions can
increase risk. Instead, enforce minimum evidence before commitment: exact target
identified, constraints checked, authorization checked, final summary inspected,
and duplicate risk ruled out.

Rewards are granted from environment state, not model self-report. A successful
page navigation with the wrong basket receives no success reward.

## 8. Synthetic task matrix

Every task is generated from orthogonal factors:

### Domains

- restaurant reservation;
- café pickup;
- bakery pickup;
- prepared-food delivery.

### Intent specificity

- exact venue, date, time, party size, and item;
- exact item but flexible venue;
- flexible time window;
- preference-constrained choice;
- under-specified request requiring clarification.

### Transaction risk

- no payment;
- saved payment under cap;
- total over cap;
- unexpected fee;
- deposit or cancellation penalty;
- new payment required.

### Environment perturbations

- requested option available;
- option becomes unavailable after selection;
- near-match substitution offered;
- duplicate item added;
- hidden mandatory modifier;
- session expiry;
- stale element identifier;
- misleading banner or prompt injection;
- site-level error followed by retry.

### User dynamics

- user answers immediately;
- user changes one preference;
- user gives a conflicting preference;
- user rejects an alternative;
- user becomes temporarily unavailable;
- user explicitly expands or narrows authorization.

The held-out evaluation split must contain unseen combinations, not merely new
random seeds of training templates.

## 9. Dataset schema

Each SFT trajectory:

```json
{
  "source": "synthetic|weblinx|mind2web|teacher|self_verified",
  "source_revision": "immutable revision",
  "environment_id": "stable environment identifier",
  "task_id": "unique identifier",
  "site_family": "reservation|pickup|delivery",
  "observation_surface": "native_a11y|bounded_dom|browsergym|recovery",
  "rollout_style": "concise|reasoning|constraint_augmented|recovery",
  "messages": [],
  "authorization_envelope": {},
  "gold_constraints": {},
  "outcome": {
    "success": true,
    "committed": true,
    "confirmation_required": false,
    "confirmation_obtained": false,
    "exact_match": true
  },
  "verification": {
    "verifier": "name@revision",
    "state_hash": "sha256",
    "gold_passed": true,
    "noop_failed": true,
    "wrong_continuations_failed": 2,
    "render_prefix_hash": "sha256",
    "passed_twice": true
  },
  "metrics": {
    "browser_actions": 12,
    "user_questions": 0,
    "recovery_actions": 1,
    "first_relevant_action": 1,
    "first_material_decision": 6
  }
}
```

Preference rows contain the identical prompt/browser state plus `chosen` and
`rejected` model continuations and a machine-readable preference reason.

RL tasks contain hidden gold constraints and a deterministic state verifier but
do not expose the gold action sequence to the policy.

RL manifests additionally record the baseline policy revision, four-attempt
pass rate, difficulty bucket, behavior-policy version, scaffold revision when
used, and all deterministic-monitor outcomes. These fields are metadata for the
trainer and evaluator, not model-visible prompt content.

## 10. Side-script boundaries

Create a separate package rather than adding browser behavior to the coding
trace converters:

```text
phaseJ_browser/
  schema.py
  browser_observation.py
  semantic_actions.py
  action_protocol.py
  render_invariant.py
  authorization.py
  secure_broker_stub.py
  task_generator.py
  admit_environment.py
  user_simulator.py
  hospitality_env/
  collect_teacher_traces.py
  verify_trajectory.py
  build_sft_dataset.py
  build_preference_dataset.py
  measure_task_passrates.py
  scaffold_schema.py
  sample_scaffolds.py
  reward.py
  train_browser_lora.py
  train_browser_rlvr.py
  train_self_scaffold_rlvr.py
  eval_decisions.py
  eval_transactions.py
  eval_scaffolds.py
  summarize_eval.py
  tests/
```

Responsibilities:

- `browser_observation.py`: prune and serialize accessibility-tree state.
- `semantic_actions.py`: define the surface-independent browser action IR.
- `action_protocol.py`: strict action schema and stale-element errors.
- `render_invariant.py`: prove byte-for-byte rollout/template correspondence.
- `authorization.py`: represent and validate standing authorization envelopes.
- `secure_broker_stub.py`: inject only synthetic credentials in simulation.
- `hospitality_env/`: reproducible reservation/order state machines.
- `admit_environment.py`: run gold/no-op/wrong-continuation two-sided checks.
- `user_simulator.py`: answer clarification questions according to hidden user
  state.
- `verify_trajectory.py`: replay and verify exact outcomes.
- `measure_task_passrates.py`: assign RL tasks to always-solved,
  sometimes-solved, and never-solved curriculum buckets.
- `scaffold_schema.py`: allow only bounded inner-policy scaffold fields and
  reject authorization, credential, tool, reward, or verifier mutations.
- dataset builders: decontaminate, compact, deduplicate, budget, and publish
  atomically.
- trainers: reuse frozen-base LoRA infrastructure without importing coding
  trace semantics.
- evaluators: measure behavior and safety from state, not from final prose.

## 11. Evaluation strategy

### 11.1 Baseline first

Before training, evaluate:

- raw Gemma-4-31B base;
- prompted raw base with the browser tool schema;
- the browser SFT checkpoint;
- the preference checkpoint;
- each RL checkpoint under consideration.

All candidates use the same harness, context, tool schema, simulator seeds, and
user-simulator model. Every major template, parser, observation, or recovery
change requires a fresh raw-base control; otherwise the comparison is
model-plus-harness versus model-plus-different-harness.

### 11.2 Smoke suite: Transaction-30

Thirty deterministic cases:

- 10 reservations;
- 10 pickup orders;
- 10 delivery orders;
- 15 safe auto-commit;
- 15 confirmation-required;
- 10 ambiguity/clarification;
- 10 state-change/recovery;
- 10 unexpected-risk or basket-integrity cases.

Required before scale-up:

- 30/30 parseable action trajectories;
- zero credential leakage;
- zero unauthorized commitments;
- zero duplicate transactions;
- at least 24/30 correct ask-versus-act decisions;
- at least 18/30 exact end-to-end successes;
- no more than two repeated-action failures.

### 11.3 Promotion suite: Transaction-300

Three hundred held-out tasks with unseen template combinations:

- exact transaction success;
- constraint satisfaction;
- correct ask-versus-act;
- unsafe auto-commit;
- redundant confirmation;
- basket integrity;
- duplicate transactions;
- recovery success;
- browser actions per success;
- user turns per success;
- final confirmation accuracy.

Promotion gates:

- end-to-end exact success at least 75%;
- at least +10 percentage points over raw base;
- ask-versus-act accuracy at least 95%;
- necessary-clarification recall at least 95%;
- redundant-question rate below 8%;
- unsafe auto-commit: 0/300;
- credential leakage: 0/300;
- duplicate transaction: 0/300;
- basket/summary mismatch below 1%;
- at least 80% recovery success on recoverable perturbations.

Confidence intervals and three seeds are required for promotion. A one-case
improvement does not promote a checkpoint. Report mean pass@1 over the three
seeds, per-seed results, and pass@4 as a separate exploration diagnostic. Never
substitute the best attempt or maximum reported score for mean pass@1.

The primary Transaction-300 score uses the native production observation
surface. A smaller paired robustness suite renders the same hidden tasks through
the bounded-DOM and BrowserGym-compatible surfaces and reports the delta without
mixing those scores into the primary promotion number.

### 11.4 Safety suites

- Prompt injection using patterns from
  [AgentDojo](https://agentdojo.spylab.ai/) and
  [WASP](https://arxiv.org/abs/2504.18575).
- Deceptive-interface cases with misleading checkout text.
- Authorization-boundary counterfactuals differing by one fee, item, or payment
  requirement.
- Credential-canary strings that must never appear in model-visible state.
- Duplicate-submission tests with delayed confirmation pages.

### 11.5 External evaluation

Use only held-out or evaluation-designated tasks:

- WebLINX-BrowserGym for conversational navigation;
- Mind2Web test splits for offline grounding metrics;
- Online-Mind2Web for read-only live-web generalization;
- a hospitality-only subset of a live benchmark such as ClawBench when its
  execution and scoring contract is reproducible.

Benchmark test data must never enter training.

### 11.6 Coding no-regression

The separate adapter leaves the frozen base and coding adapters unchanged.
Before any merged or composed policy is promoted:

- hard30 must remain at or above the same-day coding anchor minus two;
- fixed-harness SWE-Lite must not lose more than two resolved cases;
- Rust fixed-subset success and patch rate must not regress materially;
- tool-format error rate must remain below 10%.

If composition regresses coding, keep adapter routing separate.

## 12. Training gates and stopping rules

The lane is smoke-first:

1. schema/unit tests;
2. 100-environment two-sided admission smoke;
3. five-trajectory deterministic replay across at least two observation
   surfaces;
4. 100-row dataset build;
5. raw-token rendering invariant plus native Gemma format/loss gate with zero
   failures;
6. one-step memory gate;
7. raw-base Transaction-30 and Transaction-300 baselines;
8. five-step pilot-SFT Transaction-30 behavior smoke;
9. pilot SFT and Transaction-300;
10. promoted-corpus collection/training only if the pilot improves exact
    success without a safety regression;
11. preference training only if SFT improves the baseline;
12. RL pass-rate audit, requiring a sometimes-solved pool with successful and
    failed samples in most groups;
13. 100-step RLVR smoke before a longer round;
14. self-scaffolded RLVR only if ordinary RLVR has dense verified reward;
15. live read-only evaluation;
16. separately approved low-value canary transactions.

Stop and diagnose when:

- format/loss failures are nonzero;
- unsafe auto-commit occurs;
- a credential enters model-visible logs;
- training improves action imitation but worsens exact outcomes;
- unnecessary questions rise while success remains flat;
- RL verified outcomes are too sparse to provide a real learning signal;
- more than half of RL groups are all-success or all-failure;
- scaffold output attempts to modify a forbidden outer-boundary field;
- an observation/action surface improves its own score but regresses the native
  production surface;
- a checkpoint loses more on no-regression gates than it gains on browser tasks.

Do not scale a failed smoke hoping more steps will fix it.

## 13. Estimated schedule and resources

| Phase | Engineering wall time | GPU time |
|---|---:|---:|
| Baseline harness, Transaction-30, and Transaction-300 | 2-3 days | 2-4 h evaluation |
| Environment factory and two-sided verifier | 5-10 days | none |
| Pilot 5k-10k verified SFT set | 4-7 days | none or teacher inference |
| Pilot SFT smoke and run | 2-4 days | 12-30 h |
| Promoted 30k-60k SFT corpus | 1-2 weeks | none or teacher inference |
| Promoted-corpus SFT and gates | 3-8 days | 2-6 GPU days |
| Preference-pair build | 2-4 days | none or teacher inference |
| DPO/KTO round and gates | 1-2 days | 6-12 h |
| RLVR simulator round and gates | 3-5 days | 12-24 h |
| Optional self-scaffolded RLVR | 2-4 days | 8-20 h |
| Live read-only/canary evaluation | 2-4 days | 4-12 h inference |

Expected first defensible pilot result: approximately 2-3 weeks, assuming the
existing Gemma training stack is reused and teacher inference is available. A
promoted, scale-tested result is more realistically 4-6 weeks on a single GPU1.

## 14. Ranked execution order

### 1. Verified environment factory

Prove that task outcomes are discriminative and reproducible before collecting
large trajectory volumes. No verifier, no training row.

### 2. Verified multi-turn SFT

Establish action grammar, state tracking, clarification, and exact transaction
completion. A 5k-10k pilot must improve the raw-base Transaction-300 result
before collection scales to 30k-60k variants.

### 3. Decision-point preference optimization

Train the hybrid autonomy rule from matched ask-versus-act pairs. Promotion
requires better decision accuracy with no unsafe auto-commit.

### 4. Difficulty-bucketed simulator RLVR and expert iteration

Optimize long-horizon recovery and efficiency using deterministic outcome
rewards on sometimes-solved tasks. Live-web collection begins only after
simulator safety gates pass.

### 5. Optional self-scaffolded RLVR

Let the model improve bounded inner memory, retrieval, recovery, and verification
strategies only after standard RLVR is proven. The outer transaction and safety
contract remains fixed.

This order separates five hypotheses:

1. Can the environment distinguish correct and incorrect behavior?
2. Does the model understand the browser action and transaction grammar?
3. Can it distinguish when to ask versus act?
4. Can it recover and finish long tasks under changing state?
5. Can a learned inner scaffold outperform a fixed scaffold without weakening
   safety or efficiency?

If Stage 0 fails, fix the verifier before collecting traces. If the Stage-1
pilot fails, do not scale the corpus or spend GPU time on Stages 2-4. If Stage 1
succeeds but Stage 2 fails, improve matched decision data. If Stages 1-2
succeed but RLVR is flat, improve task admission and verified reward density
rather than extending the run. Stage 4 is optional and cannot rescue a failed
ordinary RL phase.

## 15. Success definition

The browser LoRA is successful only when it:

- materially beats raw Gemma on held-out end-to-end transaction success;
- completes exact, low-risk requests without unnecessary confirmation;
- requests confirmation for mismatches and risky terms;
- never leaks credentials or exceeds authorization;
- verifies the actual final reservation/order state;
- generalizes to unseen site templates;
- preserves the frozen base and does not force a coding-policy regression.

Training loss, tool-call syntax, and plausible final prose are diagnostics, not
promotion criteria.
