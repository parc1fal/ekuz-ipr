# Eight paths to a robust NBA betting agent

**An ensemble of independently seeded DQN agents is the single most effective and implementable fix for the 12-percentage-point ROI spread plaguing this system.** The root cause—Q-value overestimation amplified by a tiny replay buffer of ~15,600 transitions—means that each random seed traces a different exploration trajectory through a rugged loss landscape, landing in wildly different local optima. Averaging across K=5–10 independent Q-networks provably reduces estimator variance by a factor of 1/K, requires fewer than 50 lines of new code on top of Stable-Baselines3, and carries near-zero risk of degrading mean performance. The remaining seven directions range from theoretically elegant (distributional robustness, CVaR optimisation) to fundamentally mismatched (CFR, adversarial self-play), with offline RL via Conservative Q-Learning emerging as a strong secondary recommendation that attacks the problem's root cause at the algorithmic level.

This report evaluates each direction on five axes: mechanism, relevance to seed sensitivity, theoretical grounding, SB3 implementation difficulty (1–5 scale), and realistic expected impact. A concrete implementation sketch for the recommended approach follows.

---

## 1. Ensemble DQN averages away the variance

**What it is.** Train K independent Q-networks—either fully separate models (outer ensemble) or K heads sharing a feature trunk (Bootstrapped DQN, Osband et al., NeurIPS 2016)—and aggregate their Q-value predictions at decision time via averaging or majority vote. Averaged-DQN (Anschel, Baram & Shimkin, ICML 2017) is even simpler: save periodic snapshots of a single network during training and average their predictions at inference. MeanQ (Liang et al., ICML 2022) showed that **K=5 independent Q-networks** with independent replay sampling suffice to eliminate the need for a target network entirely, because the ensemble mean is already a low-variance target.

**How it addresses seed sensitivity.** This is the most direct statistical attack on the problem. If a single network has variance σ² across seeds, an ensemble of K independent networks has variance σ²/K. For the observed 12pp spread, K=5 should compress this to roughly **5–7pp**, and K=10 to ~4pp, assuming approximate independence of heads. The SUNRISE framework (Lee et al., ICML 2021) adds weighted Bellman backups that down-weight high-uncertainty transitions, further stabilising training. Maxmin Q-learning (Lan et al., ICLR 2020) takes the pessimistic minimum across heads, directly suppressing overestimation.

**Theoretical grounding.** Bootstrapped DQN implements approximate Thompson Sampling for deep exploration. Averaged-DQN analytically proves Target Approximation Error variance scales as 1/K. MeanQ establishes that K=5 is sufficient for stable, unbiased value estimation. These are well-cited, empirically validated results across Atari, MuJoCo, and financial trading domains.

**Implementation difficulty: 1.5/5.** The outer ensemble approach requires no modification to SB3 or the Gymnasium environment. Train K standard DQN models with different seeds, then average Q-values at inference. Averaged-DQN is even cheaper—periodically save checkpoints during a single training run and average at test time. Bootstrapped DQN with shared trunk requires subclassing SB3's `QNetwork` (~150 lines) but still needs no environment changes.

**Realistic expected results.** A **40–60% reduction in cross-seed ROI spread** is well-supported by the literature. Mean ROI should remain stable or improve slightly due to reduced overestimation bias. Training cost scales linearly with K but is trivially parallelisable. The key caveat: ensembles reduce estimator variance but cannot add signal that isn't in the data. If the underlying 15,600 transitions contain no exploitable edge, averaging over noise still yields noise.

---

## 2. Distributionally robust optimisation guards against distribution shift

**What it is.** Rather than maximising expected return under the empirical MDP, robust MDPs (Iyengar, 2005; Nilim & El Ghaoui, 2005) maximise the **worst-case expected return** over an uncertainty set of transition kernels centred on the empirical model: max_π min_{P∈U} E_P[Σγ^t r_t]. Under the (s,a)-rectangularity assumption, the robust Bellman equation holds and dynamic programming applies. Common uncertainty sets include KL-divergence balls (closed-form dual), total-variation balls (shown by Shi & Chi, JMLR 2022, to make robust MDPs *easier* to learn than standard MDPs), and Wasserstein balls.

**Relevance to sports betting.** NBA seasons differ in pace, three-point frequency, rule changes, and market efficiency. A DRO-trained agent would learn policies robust to such distributional shifts between training seasons (2010–2023) and test seasons. Wiesemann, Kuhn & Rustem (2013) introduced s-rectangularity to relax the strong independence assumption, though general non-rectangular DRO is NP-hard. Recent work by Blanchet et al. (2023) combines "double pessimism"—pessimism about data coverage *and* model uncertainty—for offline distributionally robust RL.

**However, DRO does not directly reduce seed sensitivity** in the way ensembles do. Its benefit is indirect: by regularising against fragile policies, it narrows the range of viable solutions across seeds. The "price of robustness" is well-documented—mean ROI typically decreases because the agent avoids high-EV bets that carry model uncertainty.

**Implementation difficulty: 4.5/5.** No SB3-compatible implementation exists. The RAAM library (Petrik) handles only tabular MDPs. The robust-safe-rl repository (Queeney, NeurIPS 2023) implements RAMU/OTP for continuous control but is incompatible with SB3. A BSc student would need to implement model-based robust Bellman backups with inner convex optimisation from scratch—likely 3–4 weeks of work requiring strong optimisation background.

**Realistic expected results.** A 20–40% indirect reduction in ROI spread through regularisation, but likely **negative impact on mean ROI** due to conservatism. The radius parameter ρ is difficult to tune: too large and the agent bets on nothing; too small and robustness gains vanish.

---

## 3. Counterfactual regret minimisation is the wrong paradigm

**What it is.** CFR (Zinkevich et al., NeurIPS 2007) iteratively minimises counterfactual regret at each information set in extensive-form games, converging to Nash equilibrium in two-player zero-sum games at O(1/√T). Deep CFR (Brown et al., ICML 2019) and DREAM (Steinberger et al., 2020) scale this to large games using neural function approximation.

**Why it doesn't fit.** Sports betting is a single-agent sequential decision problem against a stochastic environment, not a multi-player extensive-form game. The bookmaker sets fixed, observable odds before the agent acts—there is no alternating-move game tree. Framing the bookmaker as an adversarial opponent would compute a minimax strategy that **never exploits market inefficiencies**, guaranteeing negative ROI after vig. CFR's two-player zero-sum convergence guarantee is irrelevant when there is no strategic opponent.

**The better alternative from this family** is online learning via the Hedge algorithm (Freund & Schapire, 1997) or EXP3 (Auer et al., 2002). Hedge maintains multiplicative weights over "experts" (e.g., different seed-trained agents) and converges to the best expert's performance at O(√(T ln N)) regret. Since the bettor observes counterfactual payoffs for unbetted games, the full-information Hedge setting applies directly. This provides a **deterministic, seed-independent aggregation** layer over multiple DQN seeds with theoretical regret guarantees.

**Implementation difficulty: 5/5 for CFR** (wrong paradigm, no SB3 path). **1/5 for Hedge ensemble** (~30 lines of weight-update code on top of existing multi-seed runs).

---

## 4. CVaR optimisation targets the tail risk that reward shaping misses

**What it is.** Conditional Value at Risk at level α measures the expected return in the worst (1−α) fraction of outcomes. CVaR-constrained policy optimisation (Chow & Pavone, 2013; Tamar et al., AAAI 2015) replaces or augments the standard expected-return objective with a CVaR term, finding policies that perform well *even under bad-case realisations*. The most PPO-compatible formulation is CPPO (Ying et al., IJCAI 2022), which adds a Lagrangian CVaR constraint to the PPO clipped objective with learned VaR threshold η and dual variable λ.

**How it addresses seed sensitivity.** The 12pp spread means some seeds produce catastrophic returns. **CVaR at α=0.2 optimises the expected return conditional on being in the worst 20% of outcomes**, directly penalising high-variance policies. This complements the student's existing quadratic risk penalty, which penalises variance symmetrically (upside and downside equally). CVaR specifically targets the **left tail**—it's acceptable for some seeds to earn +6% but unacceptable for others to lose −5.7%.

**Interaction with existing reward shaping.** The quadratic risk penalty addresses second-moment variance; CVaR addresses beyond-VaR tail losses. They are complementary, not redundant. The EV-sizing bonus and edge threshold penalty remain compatible since CVaR doesn't eliminate +EV bets—it avoids overleveraging them. The student may need to reduce the quadratic penalty coefficient slightly since CVaR already discourages extreme positions.

**Implementation difficulty: 3/5.** The CPPO approach requires modifying SB3's PPO `train()` method: sort collected trajectories by return, compute empirical VaR/CVaR, add Lagrangian terms to the loss, and implement gradient updates for η and λ. A reference implementation exists (github.com/yingchengyang/CPPO) based on SpinningUp, portable to SB3 with moderate effort. A simpler alternative is reward-shaping approximation: penalise per-episode returns below a VaR threshold, implementable entirely within the Gymnasium env (difficulty 1/5, but not formally CVaR-optimal). A recent paper (arXiv:2403.06323, 2024) shows CVaR RL can be reduced to risk-neutral RL in an augmented MDP with an extra "risk budget" state variable.

**Realistic expected results.** Expected **5–8pp improvement in worst-seed ROI** and **0–3pp decrease in best-seed ROI**, compressing the 12pp spread to ~4–7pp. The critical concern is sample efficiency: CVaR policy gradients discard (1−α) of trajectory data, problematic with only ~15,600 transitions. The CPPO Lagrangian approach uses all data for the PPO loss and only adds a constraint, mitigating this issue.

---

## 5. Bayesian uncertainty enables conservative betting when data is sparse

**What it is.** Bayesian/uncertainty-aware RL maintains a distribution over Q-values rather than point estimates. MC Dropout (Gal & Ghahramani, ICML 2016) proves that dropout at test time approximates Bayesian inference: T stochastic forward passes yield mean Q̄(s,a) and variance σ²(s,a), where high variance indicates epistemic uncertainty. Randomised Prior Functions (Osband et al., NeurIPS 2018) add fixed random networks to each ensemble head (Q^k = f^k_trainable + p^k_fixed), ensuring diversity even in sparse-reward regions where bootstrapping alone generates no disagreement.

**The Kelly Criterion connection is natural and powerful.** Baker & McHale (2013, *Decision Analysis*) showed the standard Kelly criterion overestimates optimal bet size when probability estimates are uncertain. Their "Shrunken Kelly" scales bet size inversely with estimation uncertainty: f* = f_kelly × (1 − σ²/p²). Integrating MC Dropout variance directly into bet sizing creates an **uncertainty-aware action pipeline**: Q_conservative(s,a) = Q̄(s,a) − λ·σ(s,a), automatically reducing position sizes on uncertain bets. This is essentially a Lower Confidence Bound (LCB) policy—the polar opposite of UCB exploration—appropriate for exploitation under uncertainty.

**Implementation difficulty: 2/5.** MC Dropout requires only adding `nn.Dropout(p=0.1–0.3)` layers to a custom SB3 network via `policy_kwargs`, then toggling `set_training_mode(True)` at inference for T=20–50 forward passes. SB3's custom policy documentation explicitly supports this pattern. An ensemble of 5 DQN agents with different seeds provides an even simpler uncertainty estimate via disagreement, requiring no SB3 internals modification.

**Realistic expected results.** Expected ROI spread reduction to **5–8pp** with MC Dropout, **4–6pp** with full ensemble + uncertainty-penalised Kelly. Mean ROI may decrease 1–2pp for the best seeds but improve for the worst, since the agent avoids overconfident bets on poorly-estimated Q-values. The main caveat: MC Dropout uncertainty can be poorly calibrated (Verdoja & Kyrki, 2021); deep ensembles are generally better calibrated but costlier.

---

## 6. Meta-learning is elegant but starved of tasks

**What it is.** MAML (Finn, Abbeel & Levine, ICML 2017) learns an initialisation θ* that fine-tunes quickly to any new task via a few gradient steps. Each NBA season becomes a "task"; the meta-objective finds θ* that, after K inner-loop gradient updates on a season's early games, performs well on that season's remaining games. Reptile (Nichol et al., 2018) is a first-order approximation: repeatedly sample a season, train for K steps, then move the initialisation toward the adapted parameters. Reptile avoids MAML's expensive second-order gradients and is straightforward to implement.

**The 13-task bottleneck is severe.** Meta-learning typically requires 50–hundreds of tasks to learn a meaningfully general initialisation. With only 13 training seasons, the outer loop has extremely limited signal. Monthly sub-tasks (~78 tasks) could help but reduce per-task data to ~150–200 transitions, making inner-loop optimisation unreliable. A financial meta-learning paper (arXiv:2105.13599, 2021) showed improved accuracy treating each stock as a task—but that had hundreds of stocks.

**Implementation difficulty: 4/5.** SB3 has no meta-learning API. Reptile on top of SB3's PPO is feasible (~50 lines of outer-loop code using `get_parameters()`/`set_parameters()`), but full MAML requires Hessian-vector products through PPO's clipped objective, adding 3–5× computational overhead. The learn2learn library provides MAML primitives but requires extracting the policy network from SB3—significant plumbing.

**Realistic expected results.** With 13 tasks: marginal benefit, perhaps **2–3pp reduction in seed spread** and **2–4pp ROI improvement** on test seasons if fine-tuning on early test-season data is permitted. High risk of instability due to MAML's notorious hyperparameter sensitivity (inner LR, outer LR, number of inner steps, meta-batch size). The ER-MAML paper (IEEE TNNLS, 2025) notes explicitly that "existing gradient-based methods like MAML suffer from poor out-of-distribution performance due to overfitting narrow task distributions."

---

## 7. Conservative Q-Learning attacks the root cause head-on

**What it is.** CQL (Kumar et al., NeurIPS 2020, ~4,400 citations) adds a regularisation term to the standard DQN loss that **pushes down Q-values for all actions** while **pushing up Q-values for actions observed in the dataset**: L_CQL = α·E_s[log Σ_a exp(Q(s,a)) − E_{a∼D}[Q(s,a)]] + L_DQN. The net effect: out-of-distribution actions receive pessimistic Q-estimates, while in-distribution actions retain accurate values. CQL provably learns a **lower bound** on the true policy value, ensuring the agent only bets when supported by sufficient data.

**This is the most theoretically precise fix for the identified problem.** The 12pp ROI spread is a textbook symptom of offline RL's distribution shift pathology: with ~15,600 transitions, many state-action pairs are visited rarely or never. Different seeds overestimate different OOD actions, producing wildly different policies. CQL eliminates this by construction—no seed can "get lucky" with spuriously high Q-values for unseen state-action pairs.

Implicit Q-Learning (IQL, Kostrikov et al., ICLR 2022) offers an alternative that never queries OOD actions at all, using expectile regression to approximate the max-Q operator. IQL is **4× faster** than CQL and avoids the conservatism hyperparameter entirely—worth trying as a comparison.

**Implementation difficulty: 2/5.** The d3rlpy library (Seno & Imai, JMLR 2022) provides production-ready `DiscreteCQL` and `DiscreteIQL` implementations with a scikit-learn-style API. Crucially, d3rlpy has an **SB3 compatibility wrapper** (`d3rlpy.wrappers.sb3`) that converts SB3 replay buffers to d3rlpy `MDPDataset` format and wraps d3rlpy models in SB3's `predict()` interface. The main work is converting the existing replay buffer data. Key hyperparameters: `alpha` (conservative weight, try {0.5, 1.0, 5.0, 10.0}), learning rate, and target update interval.

**Realistic expected results.** Expected seed spread compression to **3–5pp** by eliminating OOD overestimation. Mean ROI may decrease slightly due to conservatism (fewer, higher-confidence bets). Risk of over-conservatism at high α values—the agent may bet on almost nothing. IQL may outperform CQL for this specific small-data regime since it avoids OOD queries entirely rather than penalising them.

---

## 8. Self-play against the bookmaker is a conceptual mismatch

**What it is.** RARL (Pinto et al., ICML 2017) frames RL as a two-player zero-sum game where an adversary applies destabilising perturbations while the protagonist learns robust responses through alternating optimisation. The idea would be to model the bookmaker as an adversary adjusting odds to minimise bettor profit.

**Why it fails for this problem.** Three fundamental issues disqualify this direction. First, the bookmaker is not an adaptive strategic opponent—odds are fixed historical data, not adversarial responses to this agent. Second, the offline setting precludes RARL's requirement for online rollouts under adversarial perturbations. Third, with only 15,600 transitions, training two competing neural networks would severely overfit. Zhang et al. (NeurIPS 2020) showed that even in simple linear-quadratic settings, RARL's alternating optimisation can destabilise the system.

A minimax policy would be maximally conservative—guaranteeing no worse than the vig, which means guaranteed negative ROI. The entire premise of sports betting is that bookmaker odds contain exploitable inefficiencies; a Nash equilibrium strategy never exploits these by construction.

**Implementation difficulty: 4.5/5.** No SB3 support for multi-agent or adversarial training. Would require building a bookmaker simulator, implementing two-agent alternating optimisation, and managing training instability—all on a tiny dataset.

**Realistic expected results.** Negligible impact on seed sensitivity (perhaps 1–2pp compression). Likely negative impact on mean ROI. **Not recommended.**

---

## The comparison at a glance

| Direction | Addresses seed sensitivity | Mean ROI impact | SB3 difficulty | Expected spread | Recommended? |
|---|---|---|---|---|---|
| **1. Ensemble DQN** | ✅ Directly (1/K variance) | Neutral to +1pp | **1.5/5** | 12→5–7pp | **✅ Primary** |
| **7. CQL/Offline RL** | ✅ Root cause (OOD fix) | −1 to −3pp | **2/5** (d3rlpy) | 12→3–5pp | **✅ Secondary** |
| **5. Bayesian/Uncertainty** | ✅ Via conservative betting | −1 to +2pp | **2/5** | 12→5–8pp | ✅ Complementary |
| **4. CVaR optimisation** | ✅ Tail risk control | −2 to +1pp | 3/5 | 12→4–7pp | ⚠️ Moderate |
| **3. Hedge/MWU** (not CFR) | ✅ Best-expert convergence | Tracks best seed | **1/5** | 12→3–5pp | ✅ Lightweight add-on |
| **2. DRO** | ⚠️ Indirect regularisation | −2 to −4pp | 4.5/5 | 12→8–10pp | ❌ Too hard |
| **6. Meta-learning** | ⚠️ Only 13 tasks | +2 to +4pp (if fine-tuning) | 4/5 | 12→9–10pp | ❌ Insufficient tasks |
| **8. Self-play** | ❌ Wrong paradigm | Likely negative | 4.5/5 | 12→10–11pp | ❌ Mismatched |

---

## Recommended implementation: ensemble DQN in one week

**Why ensemble DQN wins on all four criteria.** It (a) directly reduces seed variance through the most elementary statistical mechanism—averaging; (b) rests on well-established theory from Osband (2016) through MeanQ (2022); (c) requires minimal code changes and zero SB3 internals modification; and (d) is the only approach that reduces variance without sacrificing mean ROI. CQL is the strongest theoretical alternative but introduces a new library and risks over-conservatism; the ensemble approach dominates on feasibility.

**Concrete implementation sketch:**

**Step 1: Train K=10 independent DQN agents** (~2 days). Use the existing SB3 DQN configuration unchanged. Only vary the seed:

```python
from stable_baselines3 import DQN
seeds = list(range(10))
models = []
for s in seeds:
    model = DQN("MlpPolicy", env, seed=s, **existing_hyperparams)
    model.learn(total_timesteps=existing_timesteps)
    models.append(model)
```

**Step 2: Ensemble inference via Q-value averaging** (~50 lines). At test time, compute the mean Q-value across all K models and act greedily on the mean:

```python
import torch
def ensemble_predict(models, obs):
    q_values = [m.policy.q_net(torch.tensor(obs)) for m in models]
    mean_q = torch.stack(q_values).mean(dim=0)
    return mean_q.argmax().item()
```

**Step 3: Uncertainty-gated betting** (optional, ~30 lines). Compute Q-value standard deviation across heads. When σ exceeds a threshold, override the bet action to "skip"—this implements an LCB policy that bets only when the ensemble agrees:

```python
std_q = torch.stack(q_values).std(dim=0)
if std_q[action] > uncertainty_threshold:
    action = SKIP_ACTION  # don't bet when uncertain
```

**Step 4: Evaluate seed-variance reduction protocol.** Train the ensemble 5 times with different master seeds (each master seed generates 10 sub-seeds). Compare the ROI distribution of 5 single-agent runs vs. 5 ensemble runs. Report mean, std, min, max ROI. The key metric is **std(ROI) across master seeds**: expect a 40–60% reduction.

**Hyperparameters to tune.** Ensemble size K ∈ {3, 5, 10} (MeanQ recommends K=5 as the sweet spot). Uncertainty threshold for gating: sweep over {0.5, 1.0, 2.0} standard deviations. Aggregation method: Q-value averaging (recommended over majority voting, per Anschel 2017).

**What not to change.** The Gymnasium environment, reward shaping (quadratic risk penalty, EV-sizing bonus, edge threshold penalty), network architecture, and training hyperparameters all remain identical. The ensemble is a pure wrapper.

**Complementary quick wins.** After implementing the ensemble, two lightweight additions can stack:

- **Hedge/MWU over ensemble members** (1/5 difficulty, ~30 lines): Rather than equal-weight averaging, use multiplicative weight updates to upweight historically accurate ensemble members during the test period. This converges to the best individual agent's performance minus O(√(T ln K)/T) regret per round.
- **Averaged-DQN snapshots** (0.5/5 difficulty, ~10 lines): During each individual agent's training, save Q-network checkpoints every N target-network updates. Average the last M snapshots at inference. This is essentially free and reduces within-agent variance before ensemble averaging reduces across-agent variance.

**Five key papers to read:**

1. Osband, Blundell, Pritzel & Van Roy (2016). "Deep Exploration via Bootstrapped DQN." NeurIPS. *The foundational paper on bootstrap heads for exploration via approximate Thompson Sampling.*
2. Anschel, Baram & Shimkin (2017). "Averaged-DQN: Variance Reduction and Stabilization for Deep Reinforcement Learning." ICML. *Proves 1/K variance reduction analytically; simplest possible ensemble method.*
3. Liang, Xu, McAleer et al. (2022). "Reducing Variance in Temporal-Difference Value Estimation via Ensemble of Deep Networks." ICML (MeanQ). *Establishes K=5 as sufficient; independent sampling is key.*
4. Kumar, Zhou, Tucker & Levine (2020). "Conservative Q-Learning for Offline Reinforcement Learning." NeurIPS. *Essential reading for understanding why offline RL fails and how CQL fixes it—the theoretical complement to the ensemble approach.*
5. Henderson, Islam, Bachman et al. (2018). "Deep Reinforcement Learning that Matters." AAAI. *Documenting the seed sensitivity problem itself; provides evaluation methodology for measuring variance reduction.*

**Expected timeline.** Day 1–2: Train 10 DQN models with different seeds (parallelisable). Day 3: Implement ensemble inference and uncertainty gating. Day 4–5: Run evaluation protocol comparing single-agent vs. ensemble across 5 master seeds. Day 6–7: Tune K, uncertainty threshold, and optionally add Hedge weighting. Total: **5–7 working days**, well within the 1–2 week budget, leaving time for write-up and ablation studies.

---

## What comes next if the ensemble succeeds

If the ensemble compresses seed variance to ~5pp as expected but mean ROI remains near break-even, the signal-to-noise problem persists—the 15,600 transitions may simply not contain enough exploitable edge. Two second-stage directions then become relevant. **CQL via d3rlpy** (week 2) can be tested as an alternative training paradigm, replacing the online DQN approach entirely; it may find a more consistent policy by eliminating OOD overestimation at the source. **CVaR-constrained PPO** is the natural extension if the student shifts to PPO as the base algorithm, adding formal tail-risk control that the existing quadratic penalty approximates but doesn't guarantee. Both stack naturally on top of the ensemble: train K CQL agents in d3rlpy and average, or train K CVaR-PPO agents and average. The compounding variance reduction from ensemble + algorithmic conservatism could plausibly compress the 12pp spread below 3pp while nudging mean ROI into consistently positive territory.