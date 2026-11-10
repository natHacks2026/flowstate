# FlowState Mathematical Reference & Equation Guide

This document catalogs all physical, mathematical, neural network, and evaluation equations used across the **FlowState** Stage-1 cyber-physical attack detection pipeline.

---

## 1. Physical Mass Balance & Hydraulic Model

Stage 1 consists of the raw water supply tank ($T\text{-}101$), motorized inlet valve ($MV\text{-}101$), inlet flow meter ($FIT\text{-}101$), water level sensor ($LIT\text{-}101$), and discharge pumps ($P\text{-}101$, $P\text{-}102$).

### 1.1 Differential Mass Balance
By conservation of mass, the rate of change of water volume $V(t)$ with cross-sectional tank area $A$ is:
$$\frac{dV(t)}{dt} = A \cdot \frac{dL(t)}{dt} = Q_{\text{in}}(t) - Q_{\text{out}}(t)$$

Where:
* $L(t)$ is the tank level in millimeters ($\text{LIT101}$).
* $Q_{\text{in}}(t)$ is the volumetric inflow rate from $FIT\text{-}101$ in $\text{m}^3/\text{h}$.
* $Q_{\text{out}}(t)$ is the volumetric outflow pumped to Stage 2 in $\text{m}^3/\text{h}$.

### 1.2 Discrete Multi-Step Regression Equation
In discrete time with sampling interval $\Delta t = 5\,\text{s}$, the expected level change $\Delta L_{t \to t+1}$ is modeled via linear regression fitted on unattacked normal operational telemetry:
$$\Delta \widehat{L}_{t+1} = a \cdot \text{FIT101}_t - b \cdot \text{Pump}_t + c$$
$$\widehat{\text{LIT101}}_{t+1} = \text{LIT101}_t + \Delta \widehat{L}_{t+1}$$

Where:
* $\text{Pump}_t = \text{clip}(\text{P101\_on}_t + \text{P102\_on}_t, 0, 1)$
* $a$: Level rise coefficient per unit flow ($\text{mm} / (\text{m}^3/\text{h})$).
* $b$: Discharge drawdown per unit step while pumping ($\text{mm}/\text{step}$).
* $c$: Offset parameter absorbing steady-state losses / calibration bias.

### 1.3 Physical Mass-Balance Residual
The mass-balance residual measures the discrepancy between what the sensor reports and what physics predicted:
$$r_t = \text{LIT101}_t - \widehat{\text{LIT101}}_t$$

Normalized residual in standard deviation units ($z$-score):
$$r_{z, t} = \frac{r_t - \mu_r}{\sigma_r}$$

---

## 2. Dynamic Physical Simulator (`simulator.py`)

The simulator generates synthetic telemetry by modeling the continuous hydraulic behavior and actuator state machines.

### 2.1 Valve Travel Dynamics
The motorized valve position $v(t) \in [0.0, 1.0]$ moves with finite transit time $\tau_{\text{valve}} = 8\,\text{s}$:
$$v(t + \Delta t) = \text{clip}\left(v(t) \pm \frac{\Delta t}{\tau_{\text{valve}}}, 0.0, 1.0\right)$$

### 2.2 Inflow & Discharge Rates
$$Q_{\text{in}}(t) = \max\left(0.0, \, Q_{\text{open}} \cdot v(t) + \epsilon_{\text{flow}}\right), \quad \epsilon_{\text{flow}} \sim \mathcal{N}(0, \sigma_{\text{flow}}^2)$$
$$Q_{\text{out}}(t) = (\mathbb{I}(P_{101}) + \mathbb{I}(P_{102})) \cdot \delta_{\text{drain}}$$

### 2.3 Level Integration with Bounded Capacity
$$L_{\text{true}}(t + \Delta t) = \text{clip}\left(L_{\text{true}}(t) + k_{\text{in}} \cdot Q_{\text{in}}(t) - Q_{\text{out}}(t) + \epsilon_{\text{phys}}, \, 0.0, \, H_{\text{overflow}}\right)$$

### 2.4 Sensor Observation & Cyberattack Spoofing
The sensor reading reported to the PLC is the true physical state plus noise and malicious tampering:
$$L_{\text{sensed}}(t) = L_{\text{true}}(t) + \epsilon_{\text{sensor}} + \delta_{\text{attack}}(t)$$

Where attack injections $\delta_{\text{attack}}(t)$ take forms such as:
* **High Spoof**: $L_{\text{sensed}}(t) = H_{\text{set}} + 20 + 180 \cdot s \quad (s \in [0, 1])$
* **Low Spoof**: $L_{\text{sensed}}(t) = \max\left(0, LL_{\text{set}} - 10 - 140 \cdot s\right)$
* **Sensor Drift**: $L_{\text{sensed}}(t) = L_{\text{true}}(t) + \sum (0.5 + s) \cdot \Delta t$

---

## 3. Machine Learning Model Architecture (`model.py`)

The detector combines a 2-layer LSTM backbone, an adapter layer, and two output heads.

### 3.1 Feature Vector
At each discrete time step $t$, the input feature vector is 6-dimensional:
$$x_t = \begin{bmatrix} \text{FIT101}_{z, t} & \text{LIT101}_{z, t} & \text{MV\_open}_t & \text{P101\_on}_t & \text{P102\_on}_t & r_{z, t} \end{bmatrix}^T$$
Over a historical window of length $W = 60$ steps ($5$ minutes):
$$X = \begin{bmatrix} x_{t-W+1}, x_{t-W+2}, \dots, x_t \end{bmatrix} \in \mathbb{R}^{W \times 6}$$

### 3.2 LSTM Recurrent Transitions
For each layer $l \in \{1, 2\}$ with hidden state $h_t$ and cell state $C_t$:
$$\begin{aligned}
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) && \text{(Forget Gate)} \\
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) && \text{(Input Gate)} \\
\widetilde{C}_t &= \tanh(W_c x_t + U_c h_{t-1} + b_c) && \text{(Candidate Memory)} \\
C_t &= f_t \odot C_{t-1} + i_t \odot \widetilde{C}_t && \text{(Cell State Update)} \\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) && \text{(Output Gate)} \\
h_t &= o_t \odot \tanh(C_t) && \text{(Hidden Representation)}
\end{aligned}$$

The final representation passes through an adapter LSTM and Layer Normalization:
$$z = \text{LayerNorm}\left(h_T^{(\text{adapter})}\right) \in \mathbb{R}^{32}$$

### 3.3 Dual Task Heads
1. **Hybrid Physics-Guided Forecast Head**:
   Predicts next-step flow and a learned neural residual correction to the physical mass balance:
   $$\widehat{\text{FIT101}}_{t+1} = W_{\text{fit}} z + b_{\text{fit}}$$
   $$\widehat{\text{LIT101}}_{t+1} = \text{LIT101}_{\text{physics}, t+1} + \left(W_{\text{lit}} z + b_{\text{lit}}\right)$$

2. **Classification (Attack) Head**:
   $$\text{logit} = W_2 \cdot \text{GELU}(W_1 z + b_1) + b_2$$
   $$p_{\text{attack}} = \sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}$$

---

## 4. Multi-Task Training Loss Function

The model is trained using a composite loss function:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{forecast}} + \mathcal{L}_{\text{classification}}$$

### 4.1 Huber Forecasting Loss
Robust to occasional sensor measurement spikes, applied only during normal operation:
$$\mathcal{L}_{\text{forecast}} = \frac{1}{N_{\text{normal}}} \sum_{i \in \text{normal}} \left( \ell_{\delta}(\widehat{\text{FIT}}_{i}, \text{FIT}_{i}) + \ell_{\delta}(\widehat{\text{LIT}}_{i}, \text{LIT}_{i}) \right)$$
$$\ell_{\delta}(a, y) = \begin{cases} 
\frac{1}{2}(a - y)^2 & \text{for } |a - y| \le \delta \\
\delta \cdot \left(|a - y| - \frac{1}{2}\delta\right) & \text{otherwise}
\end{cases} \quad (\delta = 1.0)$$

### 4.2 Class-Imbalance-Weighted Binary Cross-Entropy
$$\mathcal{L}_{\text{classification}} = -\frac{1}{N_{\text{unmasked}}} \sum_{j \in \text{unmasked}} \left[ w_{\text{pos}} \cdot y_j \log(p_j) + (1 - y_j) \log(1 - p_j) \right]$$
$$w_{\text{pos}} = \text{clip}\left(\frac{N_{\text{negative}}}{N_{\text{positive}}}, \, 1.0, \, 50.0\right)$$

---

## 5. ICS Deterministic Invariant Rules (`run_experiments.py`)

Physical systems possess hard boundary invariants that cannot be violated under legitimate operation:

1. **Auxiliary Pump Rule**:
   Pump $P\text{-}102$ is a cold standby backup and never runs in standard plant cycles:
   $$I_{\text{P102}}(t) = \mathbb{I}(\text{P102}_t = 2)$$

2. **Causal Premature Pump Trip / Lockout Rule**:
   Normal drainage requires $P\text{-}101$ to run for at least $500\,\text{s}$ ($\min_{\text{normal}} = 659\,\text{s}$). If $P\text{-}101$ aborts early:
   $$I_{\text{abort}}(t) = \begin{cases} 
   1 & \text{if pump turned OFF at } t \text{ and } \Delta t_{\text{run}} < 500\,\text{s} \\
   0 & \text{otherwise}
   \end{cases}$$
   The lockout persists forward during the rest cycle:
   $$I_{\text{lockout}}(t) = \begin{cases} 
   1 & \text{if in lockout state and } \text{P101}_t = 1 \\
   0 & \text{if pump restarts legitimately}
   \end{cases}$$

### 5.1 Ensembled Anomaly Score
The overall score blends machine learning probabilities with deterministic invariants:
$$S(t) = \max\left(p_{\text{attack}}(t), \, I_{\text{P102}}(t), \, I_{\text{lockout}}(t)\right)$$

An alert triggers when:
$$\text{Alert}(t) = \mathbb{I}(S(t) \ge \tau)$$

---

## 6. Threshold Optimization & Validation Metrics (`evaluate.py`)

### 6.1 Constrained Threshold Selection
Given validation scores $S_{\text{val}}$ and a maximum allowable false alarm budget $B_{\text{FA}}$ (e.g. $0.5$ or $2.0\,\text{FA/day}$):
$$\tau^* = \min \left\{ \tau \in \mathcal{C} \;\middle|\; \text{FPR}(\tau) \le 0.02 \;\land\; \text{FA}_{\text{day}}(\tau) \le B_{\text{FA}} \right\}$$
Where $\mathcal{C}$ is evaluated in **descending order** from the highest quantile downwards.

### 6.2 Performance Metrics
1. **Point-wise Classification**:
   $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

2. **Event-Based False Alarm Rate**:
   A false alarm is an unbroken contiguous run of positive predictions $[\tau_s, \tau_e]$ outside any attack or 15-minute grace period:
   $$\text{FA}_{\text{day}} = \frac{\sum_{k=1}^K \mathbb{I}\left(\text{Run}_k \cap \text{Attacks}_{\text{grace}} = \emptyset\right)}{T_{\text{normal}} \text{ (in days)}}$$
