# Aurora Research Methodology

## Primary Research Hypothesis

Financial markets exhibit hidden regime structures. Different predictive models perform
optimally under different market conditions. A dynamic ensemble conditioned on latent
market regimes can outperform static architectures across:

- **Forecast accuracy**: MSE, RMSE, MAE, MAPE, directional accuracy
- **Trading performance**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Risk-adjusted returns**: max drawdown, CVaR, regime-specific attribution

---

## Regime Detection

### Gaussian Hidden Markov Model

Let $\mathbf{x}_t \in \mathbb{R}^d$ be the observation vector at time $t$, and
$s_t \in \{1, \ldots, K\}$ the latent regime. The model assumes:

$$P(s_t = j \mid s_{t-1} = i) = A_{ij}$$

$$P(\mathbf{x}_t \mid s_t = k) = \mathcal{N}(\mathbf{x}_t; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

Parameters $(A, \boldsymbol{\mu}, \boldsymbol{\Sigma})$ are learned via the
Baum-Welch (EM) algorithm. The Viterbi algorithm decodes the most likely
regime path $\hat{s}_{1:T}$.

### Markov Switching Autoregression

Hamilton's (1989) Markov Switching AR(p) model for returns $r_t$:

$$r_t = \mu_{s_t} + \sum_{i=1}^{p} \phi_{i,s_t} r_{t-i} + \sigma_{s_t} \varepsilon_t$$

where $\varepsilon_t \sim \mathcal{N}(0,1)$ and the variance $\sigma^2_{s_t}$
switches across regimes, capturing volatility clustering.

### Ensemble Regime Assignment

The final regime label is determined by:

$$\hat{s}_t = \begin{cases}
\arg\max_k P_{\text{HMM}}(s_t = k \mid \mathbf{x}_{1:t}) & \text{if } \max_k P_{\text{HMM}} \geq \tau_c \\
\arg\max_k P_{\text{MS}}(s_t = k \mid r_{1:t}) & \text{otherwise}
\end{cases}$$

where $\tau_c = 0.6$ is the confidence threshold.

---

## Regime-Adaptive Ensemble

### Model Weighting

For each regime $k$, model weights $w^{(k)}_m$ are learned from walk-forward RMSE:

$$w^{(k)}_m = \frac{\text{softmax}(-\text{RMSE}^{(k)}_m / \hat{\sigma})}{\sum_{m'} \text{softmax}(-\text{RMSE}^{(k)}_{m'} / \hat{\sigma})}$$

with minimum weight clipping at $w_{\min} = 0.05$ to prevent degenerate solutions.

### Confidence Scaling

When regime confidence $c_t < \tau_c$, weights are interpolated toward uniform:

$$\tilde{w}^{(k)}_m = c_t \cdot w^{(k)}_m + (1 - c_t) \cdot \frac{1}{M}$$

This gracefully handles ambiguous market states.

### Final Prediction

$$\hat{y}_{t+h} = \sum_{m=1}^{M} \tilde{w}^{(\hat{s}_t)}_m \cdot \hat{y}^{(m)}_{t+h}$$

---

## Uncertainty Quantification

### Transformer Quantile Regression

The pinball (quantile) loss for quantile $q$:

$$\mathcal{L}_q(\hat{y}, y) = \mathbb{E}\left[\max(q(y - \hat{y}),\; (q-1)(y - \hat{y}))\right]$$

Training with $q \in \{0.1, 0.25, 0.5, 0.75, 0.9\}$ produces a full predictive
distribution without distributional assumptions.

### MC-Dropout

At inference, LSTM/GRU models retain dropout (enabling Monte Carlo sampling):

$$\text{Var}[\hat{y}] \approx \frac{1}{T} \sum_{t=1}^{T} \hat{y}_t^2 - \left(\frac{1}{T} \sum_{t=1}^{T} \hat{y}_t\right)^2$$

with $T = 50$ forward passes.

---

## Feature Engineering

### Hurst Exponent (R/S Analysis)

$$H = \lim_{n \to \infty} \frac{\log(\mathbb{E}[R(n)/S(n)])}{\log(n)}$$

- $H = 0.5$: random walk (Brownian motion)
- $H > 0.5$: persistent / trending
- $H < 0.5$: anti-persistent / mean-reverting

### Garman-Klass Volatility Estimator

$$\sigma^2_{\text{GK}} = 0.5 \ln\left(\frac{H}{L}\right)^2 - (2\ln 2 - 1)\ln\left(\frac{C}{O}\right)^2$$

More efficient than close-to-close estimators; uses full OHLC information.

---

## Backtesting Framework

### Transaction Cost Model

Net return at time $t$:

$$r^{\text{net}}_t = w_{t-1} r^{\text{price}}_t - \delta_w (c_{\text{comm}} + c_{\text{slip}})$$

where $\delta_w = |w_t - w_{t-1}|$ is the signed weight change (turnover).

### Sharpe Ratio

$$\text{SR} = \frac{\sqrt{252} \cdot \mathbb{E}[r_t - r_f]}{\text{Std}[r_t - r_f]}$$

### Maximum Drawdown

$$\text{MDD} = \min_{t \leq T} \frac{V_t - \max_{s \leq t} V_s}{\max_{s \leq t} V_s}$$

---

## Statistical Validation

### Diebold-Mariano Test

Test $H_0$: equal predictive accuracy between models 1 and 2.

$$d_t = L(e^{(1)}_t) - L(e^{(2)}_t), \quad L(\cdot) = (\cdot)^2 \text{ or } |\cdot|$$

$$\text{DM} = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d})}} \sim t_{T-1}$$

where $\hat{V}$ uses Newey-West HAC variance estimation for $h$-step ahead forecasts.
