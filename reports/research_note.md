# Weight-Space Interventions for Swedish Language Adaptation

**Md Zubayer Ahmed Fahim**

Summary of the final research note.

## 1. Research question and motivation

Can selected weight differences preserve Swedish improvement while limiting English
deterioration? I wanted to investigate whether checkpoint differences could identify
updates worth keeping without another training run, and whether these interventions
would outperform simpler alternatives. This matters to SE.LLMA because adaptation
also affects a model's existing capabilities.

## 2. Critical assessment of Weight-Space Programming

The independently checked copying-head experiment in *Sparks of Weight-Space
Programming* [1] was particularly convincing. Larger-model edits relied more on
behavioural feedback, and not every attempt succeeded. This motivated a narrower
test of whether numerical patterns in checkpoint differences can guide useful
interventions, rather than a claim to understand Swedish competence from raw weights.

## 3. Hypothesis and experimental design

I compared SmolLM2-135M with a compatible Swedish-adapted checkpoint [4]. My hypothesis
was that allocating more of a fixed 10% or 30% update budget to layers with larger
relative changes would preserve more Swedish improvement than random selection or
DARE. Lower Swedish negative log-likelihood (NLL) would support the claim; equal or
higher NLL would challenge it. English NLL checked regression, without a predefined
acceptable-loss threshold.

Controls included random, global magnitude, fixed-count DARE and layer-random selection.
After an eight-text AI-authored pilot per language, I evaluated unchanged methods on
150 frozen Wikipedia excerpts per language. Every method used the same texts and
random controls used seeds 17, 29 and 43. Revisions, document losses and settings are
recorded in the repository. Compatibility checks passed, but conflicting ancestry
information [4, 5] prevents attributing every checkpoint difference solely to Swedish
training.

## 4. Results and follow-up investigations

The main Wikipedia comparison gave these NLL values; lower is better. Random and
DARE entries average the fixed seeds.

| Method | Swedish NLL | English NLL |
|---|---:|---:|
| Base | 4.146 | 2.873 |
| Swedish checkpoint | 2.132 | 3.712 |
| Random 30% | 4.171 | 3.544 |
| Magnitude 30% | 3.966 | 4.313 |
| Layer-aware 30% | 4.332 | 4.289 |
| DARE 30% | 22.196 | 22.932 |

The layer-aware rule lost to random at both budgets. Magnitude 30% retained only 8.9%
of the full Swedish gain. DARE's large losses concern this checkpoint difference and
rescaling setup, not a general failure of DARE [2, 3].

**Balanced allocation.** I next asked whether distributing retained updates more
evenly would help. Keeping the largest 30% within each layer preserved the overall
budget. On 50 new articles per language, balanced Swedish NLL was 4.049 versus 4.371
for the original rule. The improvement was 0.322, with a paired 95% interval of
0.268 to 0.380. However, it retained only 7.9% of the full gain and English NLL was
4.412. The 10% comparison was unclear. This raised the question of whether removing
individual updates was itself too disruptive.

**Dense adaptation strength.** I then kept every update but scaled its strength by
α. Eight fixed values were compared on legacy development texts, selecting the
lowest equal-language mean NLL. Selection chose α = 1.0 before another 50-article-
per-language confirmation. This full endpoint scored 2.131/3.698 on Swedish/English,
versus 3.997/4.326 for balanced 30% and 4.148/2.871 for base. English still regressed
relative to base. No improved intermediate strength was discovered; other α values
were evaluated only on development data.

## 5. Discussion and limitations

The original hypothesis was unsupported. Balanced allocation improved one comparison
but did not resolve the English trade-off. Coordinated changes across many weights
are one possible explanation, but the experiments do not establish that mechanism.

One checkpoint pair, short Wikipedia excerpts and possible training-data overlap
limit generalization. NLL measures text prediction. Matching parameter counts does
not match update magnitudes; no English tolerance was fixed, and ancestry remains
uncertain. These results do not rule out other weight-space interventions.

## 6. Recommendations and future research

I would retain this as a diagnostic study rather than deploy the tested sparse rule.
I would first verify the parent checkpoint, then compare magnitude-matched edits
with a development-set English tolerance and evaluate another Swedish domain,
including grammar or idiomatic language. A future question is whether an AI reasoning
model could use weight-level evidence to outperform simple selection rules.

## References

[1] [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).
[2] [Yu et al., DARE](https://arxiv.org/abs/2311.03099).
[3] [Deng et al., DARE the Extreme](https://arxiv.org/abs/2410.09344).
[4] [Swedish checkpoint](https://huggingface.co/jekunz/smollm-135m-cpt-fineweb-swedish).
[5] [Kunz, Idiomatic Language Acquisition](https://arxiv.org/abs/2602.03484).
[6] [SE.LLMA](https://wasp-sweden.org/research/projects/sellma/).

### AI Tool Usage

AI tools were used to assist with debugging, test code generation, example codes and preliminary review of experimental results. Generated suggestions were evaluated for technical correctness and consistency with the recorded findings.
