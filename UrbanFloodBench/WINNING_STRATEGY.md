# UrbanFloodBench — Winning Strategy (Top 1%)

> **Author:** GitHub Copilot (Claude Sonnet 4.6)  
> **Date:** March 4, 2026 | Updated: March 6, 2026  
> **Goal:** #1 on the leaderboard. Every architectural and training decision below is justified against the metric and data structure.

---

## 0. TL;DR — The Winning Recipe

```
Spatio-Temporal Heterogeneous GNN
  + Residual / delta prediction (predict change in WL, not absolute WL)
  + Scheduled sampling (curriculum autoregressive training)
  + Physics-informed post-processing (elevation floor)
  + Separate per-model training with shared backbone
  + Ensemble: GNN + LightGBM per-node baseline
```

---

## 1. Metric Strategy — Where Points Come From

### Scoring hierarchy (bottom-up):
```
RMSE per node
  → mean over nodes of same type  ← 1D and 2D get EQUAL weight here
  → mean(1D score, 2D score)      = event score
  → mean over all test events     = model score
  → mean(Model1 score, Model2 score) = FINAL
```

### Critical implications for architecture:

| Observation | What it means |
|---|---|
| 1D and 2D get equal weight | Don't let 2D (huge node count) dominate training loss — use **weighted loss** or **separate models** |
| Each event weighted equally | A catastrophically wrong event (e.g., rare extreme rainfall) tanks the score — need **robust generalization** |
| Each model weighted equally | Model 1 has only 17 1D nodes; Model 2 has 198 — the architecture must handle both scales |
| Standardized by std_dev | Getting the **dynamic range** right matters more than absolute precision |
| Warmup=10 timesteps excluded | We can use warmup for state initialization — **always use it** |

### Loss function design:
```python
loss = 0.5 * MSE(pred_1d, target_1d) / var_1d  +  0.5 * MSE(pred_2d, target_2d) / var_2d
```
This mirrors the evaluation metric exactly. Compute `var_1d` and `var_2d` from training data per model.

---

## 2. Data Structure — What We Actually Have

### Confirmed schemas from local data:

```
Models/
  Model_1/
    train/
      1d_nodes_static.csv        [17 rows]  cols: node_idx, position_x, position_y, depth,
                                                   invert_elevation, surface_elevation, base_area
      2d_nodes_static.csv      [3716 rows]  cols: node_idx, position_x, position_y, area,
                                                   roughness, min_elevation, elevation, aspect,
                                                   curvature, flow_accumulation
      1d_edges_static.csv        [16 rows]  cols: edge_idx, relative_position_x, relative_position_y,
                                                   length, diameter, shape, roughness, slope
      2d_edges_static.csv      [7935 rows]  cols: edge_idx, relative_position_x, relative_position_y,
                                                   face_length, length, slope
      1d_edge_index.csv          [16 rows]  cols: edge_idx, from_node, to_node
      2d_edge_index.csv        [7935 rows]  cols: edge_idx, from_node, to_node
      1d2d_connections.csv       [16 rows]  cols: connection_idx, node_1d, node_2d
      event_X/
        timesteps.csv                       cols: timestep_idx, timestamp
        1d_nodes_dynamic_all.csv            cols: timestep, node_idx, water_level, inlet_flow
        2d_nodes_dynamic_all.csv            cols: timestep, node_idx, rainfall, water_level, water_volume
        1d_edges_dynamic_all.csv            cols: timestep, edge_idx, flow, velocity
        2d_edges_dynamic_all.csv            cols: timestep, edge_idx, flow, velocity
  Model_2/  (same structure)
    1d_nodes_static.csv        [198 rows]
    2d_nodes_static.csv       [4299 rows]
    1d_edge_index.csv          [197 rows]
    2d_edge_index.csv         [9876 rows]
    1d2d_connections.csv       [197 rows]
    train: 69 events
    test:  30 events
```

### Test event structure:
- First 10 timesteps: ALL dynamic variables provided (`water_level`, `inlet_flow`, `rainfall`, `water_volume`, `flow`, `velocity`)
- Timesteps 11+: Only `rainfall` is provided — everything else is NaN / must be predicted
- Warmup timesteps (0–9) are **excluded from scoring**

### Dataset sizes at a glance:
| | Model 1 | Model 2 |
|---|---|---|
| 1D nodes | 17 | 198 |
| 2D nodes | 3,716 | 4,299 |
| 1D edges | 16 | 197 |
| 2D edges | 7,935 | 9,876 |
| 1D-2D links | 16 | 197 |
| Train events | 68 | 69 |
| Test events | 29 | 30 |
| Timestep interval | 5 min | 5 min |
| Typical event length | ~94 steps (~8 hrs) | varies |

---

## 3. Why GNN is the Right Architecture

### The graph topology matters physically:
- **1D network**: Tree/DAG topology of pipes — water flows downstream following gravity (slope, invert elevation)
- **2D mesh**: Regular grid of surface cells — flood wave propagates radially from rainfall zones
- **1D-2D coupling**: Manholes/drains transfer water between surface and subsurface — bidirectional

### Why simple architectures will fall short:
| Approach | Problem |
|---|---|
| Independent per-node MLP | Ignores spatial propagation entirely |
| Global LSTM | Can't scale to 4000+ nodes, no structure |
| CNN on 2D grid | Grid is irregular (different cell sizes/areas) |
| GBM per node | Doesn't model spatial coupling; pure tabular lag features |
| Pure physics simulation | We're replacing it, not replicating it |

### Why GNN wins:
1. **Message passing = physical flow propagation** — water level at node i at time t+1 depends on its neighbors at time t
2. **Heterogeneous edges** (1D pipe, 2D surface, 1D-2D coupling) each carry different physics
3. **Scales gracefully** from 17 to 4299 nodes without architectural changes
4. **Static features as node/edge attributes** — naturally integrated

---

## 4. Architecture Design

### 4.1 Full Architecture: Heterogeneous GNN + GRU (HeteroGNN-GRU)

```
For each timestep t:

INPUT:
  x_1d[t]  = [static_1d | water_level[t-1], inlet_flow[t-1]]   shape: (N_1d, F_1d)
  x_2d[t]  = [static_2d | rainfall[t], water_level[t-1], water_volume[t-1]]  shape: (N_2d, F_2d)
  e_1d[t]  = [static_1d_edge | flow[t-1], velocity[t-1]]       shape: (E_1d, F_e1d)
  e_2d[t]  = [static_2d_edge | flow[t-1], velocity[t-1]]       shape: (E_2d, F_e2d)
  e_12[t]  = [connection features]                              shape: (E_12, F_e12)

STEP 1 — Node/Edge Embedding:
  h_1d = Linear(x_1d)  + LayerNorm                             (N_1d, D)
  h_2d = Linear(x_2d)  + LayerNorm                             (N_2d, D)
  he_1d = Linear(e_1d) + LayerNorm                             (E_1d, D)
  he_2d = Linear(e_2d) + LayerNorm                             (E_2d, D)

STEP 2 — Heterogeneous GNN (2-3 rounds of message passing):
  For each edge type (1d→1d, 2d→2d, 1d→2d, 2d→1d):
    messages = edge_mlp(concat(h_src, h_dst, he_edge))
    aggr = scatter_mean(messages, dst_node)
  h_1d = h_1d + aggr_1d_to_1d + aggr_2d_to_1d    (residual)
  h_2d = h_2d + aggr_2d_to_2d + aggr_1d_to_2d    (residual)
  Apply LayerNorm after each round

STEP 3 — Temporal Module (GRU per node):
  state_1d[t] = GRU_1d(h_1d[t], state_1d[t-1])              (N_1d, D_hidden)
  state_2d[t] = GRU_2d(h_2d[t], state_2d[t-1])              (N_2d, D_hidden)

STEP 4 — Prediction Head (predict DELTA, not absolute):
  delta_wl_1d[t] = MLP_1d(state_1d[t])                       (N_1d, 1)
  delta_wl_2d[t] = MLP_2d(state_2d[t])                       (N_2d, 1)

  wl_1d[t] = wl_1d[t-1] + delta_wl_1d[t]    ← RESIDUAL PREDICTION
  wl_2d[t] = wl_2d[t-1] + delta_wl_2d[t]

OUTPUT:
  water_level predictions for all nodes at timestep t
  → fed back as input for t+1 (autoregressive)
```

### 4.2 Why Delta (Residual) Prediction is Critical

Water levels change slowly between 5-minute steps. The delta is a small number close to 0. This:
- Makes the learning problem easier (smaller output scale)
- Reduces error accumulation in autoregressive rollout
- Naturally handles the "baseline" water level at each node (invert elevation + typical depth)

### 4.3 Hyperparameters (suggested starting point)
```python
D           = 128       # node hidden dim
D_hidden    = 128       # GRU hidden dim
n_gnn_layers = 2        # GNN message passing rounds
n_mlp_layers = 2        # MLP layers in heads
dropout      = 0.1
lr           = 3e-4
batch_size   = 1 event  # process one event at a time
warmup_steps = 10       # use to initialize GRU state
```

---

## 5. Training Strategy

### 5.1 Scheduled Sampling (Critical for Autoregressive Stability)

Naive teacher forcing (always using ground-truth WL as input) causes train/test mismatch.

```python
# Curriculum: start with teacher forcing, gradually use own predictions
p_teacher = max(0.0, 1.0 - epoch / 50)   # linear decay over 50 epochs

for t in range(10, T):
    if random() < p_teacher:
        input_wl = gt_wl[t-1]    # teacher forcing
    else:
        input_wl = pred_wl[t-1]  # own prediction

    pred_wl[t] = model(input_wl, rainfall[t], static_feats, hidden_state)
```

### 5.2 Loss Function

```python
def loss(pred_1d, pred_2d, gt_1d, gt_2d, std_1d, std_2d):
    # Only compute on timesteps > 10 (warmup excluded)
    mse_1d = ((pred_1d - gt_1d) ** 2).mean()
    mse_2d = ((pred_2d - gt_2d) ** 2).mean()
    # Standardize to match eval metric
    return 0.5 * (mse_1d / std_1d**2) + 0.5 * (mse_2d / std_2d**2)
```

Compute `std_1d` and `std_2d` from training water_level values per model.

### 5.3 Multi-Step Loss (Reduce Drift)

Beyond single-step prediction, compute loss on k-step rollouts:

```python
total_loss = sum(weight[k] * loss_at_step(t+k) for k in [1, 5, 10, 20])
```

Initially train with k=1 only, then gradually introduce multi-step losses.

### 5.4 Training Protocol
1. **Phase 1** (epochs 1–20): Teacher forcing, single-step loss, lr=3e-4
2. **Phase 2** (epochs 21–50): Scheduled sampling decay, k=1..5 multi-step loss
3. **Phase 3** (epochs 51–80): 0% teacher forcing, full multi-step loss, lr=1e-4
4. **Fine-tune** (epochs 81–100): Train on full event rollouts, very low lr=1e-5

---

## 6. Feature Engineering

### 6.1 Key Derived Static Features

For 1D nodes:
- `fill_ratio_capacity` = (surface_elevation - invert_elevation) / depth  → surcharge risk
- `relative_depth` = normalization of depth across all 1D nodes per model
- Degree (number of connected 1D edges and 1D-2D connections)

For 2D nodes:
- `terrain_slope_proxy` = (elevation - min_elevation) / sqrt(area)
- `catchment_potential` = flow_accumulation × area  → upstream drainage volume proxy
- Distance to nearest 1D node (from 1D-2D connections)

For edges:
- `hydraulic_gradient` = slope × length  → flow potential
- For 2D edges: `unit_width_flow` capacity = face_length × slope

### 6.2 Temporal Input Features at Each Step
- Previous water_level (t-1), delta_wl (t-1 minus t-2), delta_wl (t-2 minus t-3)  → 3 lag features
- Current rainfall[t] (always known)
- Cumulative rainfall over last 6, 12, 24 steps  → antecedent moisture proxy
- Time of event (normalized step index)  → captures event phase

---

## 7. Specific Tricks for This Competition

### 7.1 Warmup State Initialization (Do NOT Ignore This)

10 timesteps of ground-truth are given. Run the GRU forward through those 10 steps with ground truth inputs to get a meaningful initial hidden state. This is like a "warm start" for the autoregressive rollout.

```python
# Initialize hidden state using warmup
hidden = None
for t in range(10):
    hidden = model.gru_step(true_features[t], hidden)
# Now roll out autoregressively from t=10
pred = []
for t in range(10, T):
    out, hidden = model.step(pred[-1] if pred else true_wl[9], rainfall[t], hidden)
    pred.append(out)
```

### 7.2 Elevation-Based Physical Clipping

For 1D nodes: water_level cannot be below invert_elevation (bottom of pipe).
For 2D nodes: water_level cannot be below min_elevation (terrain floor).

```python
wl_1d = torch.maximum(pred_1d, invert_elevation_1d)
wl_2d = torch.maximum(pred_2d, min_elevation_2d)
```

Apply this DURING autoregressive rollout AND on final predictions. Violations add to RMSE unnecessarily.

### 7.3 Per-Model Training with Shared Backbone

Train a shared model backbone but with model-specific normalization statistics and output heads. This:
- Leverages cross-model generalization (137 events total vs ~69 per model)
- Adapts to each model's scale via model-specific layer norm statistics

### 7.4 Event-Difficulty Weighting

Some events have more extreme rainfall and larger water level fluctuations — higher std means harder to predict. Monitor per-event validation RMSE during training. Upsample difficult events.

### 7.5 Test-Time Augmentation / Calibration

After generating predictions from the autoregressive model, apply a **per-node bias correction** using the warmup steps:
```python
# Compare model prediction vs ground truth at warmup steps
bias = mean(gt_wl[0:10] - pred_wl[0:10])  # per node
# Apply correction to future predictions
pred_wl[10:] += bias
```
This corrects any systematic per-node offset.

---

## 8. LightGBM Baseline (Parallel Fast Track)

Build this immediately — it gives a score on the leaderboard within hours of the competition start.

### Feature construction per (node, timestep) sample:
```
Static features of the node
+ Static features of connected neighbors (aggregated: mean, max)
+ water_level[t-1], water_level[t-2], water_level[t-3]
+ delta_wl[t-1] = wl[t-1] - wl[t-2]
+ rainfall[t], cumrain_6step, cumrain_12step
+ timestep (position in event)
+ event total rainfall (global event feature)
```

**Target:** `water_level[t]`

**Training:** One row per (node, timestep, event). Train separate GBMs for 1D and 2D nodes, separate models for Model 1 and Model 2.

**Test prediction:** Autoregressively predict t=10, use as input for t=11, etc.

**Expected performance:** ~0.3–0.5 Standardized RMSE (decent baseline, not winning)

---

## 9. Ensemble Strategy

| Model | Weight | Rationale |
|---|---|---|
| HeteroGNN-GRU (best checkpoint) | 0.50 | Best structural understanding |
| HeteroGNN-GRU (different seed/HP) | 0.25 | Diversity |
| LightGBM per-node | 0.15 | Orthogonal approach, catches node-level patterns |
| Simple persistence + rainfall correction | 0.10 | Robust fallback for stable nodes |

Use **weighted average** for final prediction. Tune weights on a held-out validation event.

---

## 10. Implementation Roadmap

### Phase 1: Foundation (Days 1–2)
- [ ] Fix EDA notebook to load local data correctly with proper schemas
- [ ] Compute and store `std_1d`, `std_2d` per model (evaluation normalization constants)
- [ ] Analyze event length distribution, rainfall distributions, WL distributions per model
- [ ] Verify graph connectivity — visualize with networkx or geopandas + shapefiles

### Phase 2: Baseline (Days 2–3)
- [ ] Build LightGBM per-node baseline with lag features
- [ ] Autoregressive inference loop for LightGBM
- [ ] Generate first submission — understand the leaderboard scale

### Phase 3: GNN Architecture (Days 3–6)
- [ ] Implement heterogeneous graph construction (PyTorch Geometric `HeteroData`)
- [ ] Implement HeteroGNN-GRU with residual delta prediction
- [ ] Teacher forcing training loop with metric-aligned loss
- [ ] Validate on held-out events (leave 10 events out per model)

### Phase 4: Training Refinement (Days 6–10)
- [ ] Add scheduled sampling
- [ ] Add multi-step rollout loss
- [ ] Physical constraint clipping
- [ ] Warmup-based state initialization + bias correction

### Phase 5: Ensemble + Polish (Days 10–14)
- [ ] Train multiple GNN seeds/configs
- [ ] Tune ensemble weights on local held-out events
- [ ] Verify submission format exactly matches sample_submission
- [ ] Final submission with best ensemble

---

## 11. Tools & Libraries

```python
# Graph ML
torch_geometric          # HeteroData, SAGEConv, GATConv, to_hetero
torch                    # base framework

# Tabular baseline
lightgbm
xgboost

# Data
pandas, numpy
pyarrow                  # parquet output

# Visualization / debug
matplotlib, seaborn
networkx                 # graph analysis
geopandas                # shapefile visualization (optional)

# Kaggle environment
torch >= 2.0
torch-geometric >= 2.4
```

---

## 12. Red Flags to Monitor

| Risk | Mitigation |
|---|---|
| Autoregressive error drift | Scheduled sampling + multi-step loss |
| 1D nodes dominating 2D (or vice versa) | Metric-aligned loss weighting (0.5/0.5) |
| Model 1 vs Model 2 scale mismatch | Per-model normalization statistics |
| Overfitting to train events | Leave-event-out CV; regularization dropout |
| Physical violations (WL < invert elev) | Clipping during rollout |
| Warmup leakage | Strictly exclude t<10 from loss computation |
| OOM on Kaggle (GPU 16GB) | Use event-by-event processing; 2D graph has 7935 edges - manageable |

---

## 13. Validation Protocol

**Never use test events for any decision.** Split per model:
- Training: all events except last 10 in chronological order
- Validation: last 10 events (or random 10 events) per model

Compute validation Standardized RMSE matching the exact leaderboard formula:
```python
def standardized_rmse(pred, gt, std):
    return np.sqrt(((pred - gt) ** 2).mean()) / std

# Per event:
rmse_1d = mean over nodes of standardized_rmse(pred_1d_node, gt_1d_node, std_1d)
rmse_2d = mean over nodes of standardized_rmse(pred_2d_node, gt_2d_node, std_2d)
event_score = 0.5 * (rmse_1d + rmse_2d)

# Per model:
model_score = mean(event_score for event in test_events)

# Final:
final = 0.5 * (model1_score + model2_score)
```

---

## 14. Data Preprocessing & Feature Selection Plan (Phase 1 Prerequisite)

> This section was added March 6, 2026, based on detailed EDA of the actual data.
> All decisions below are data-driven. Key statistics observed:
> - Model 1: 1D WL std=16.87 ft, 2D WL std=14.37 ft; range ~[286, 348] ft
> - Model 2: 1D WL std=2.52 ft, 2D WL std=2.80 ft; range ~[25, 55] ft  ← completely different scale
> - Delta WL (1D, M1): mean=−0.019 ft, std=0.202 ft, max=3.12 ft → ~84× smaller than absolute WL
> - `invert_elevation` correlates 0.9997 with mean 1D WL in M1, 0.805 in M2 → **dominant baseline signal**
> - `shape` column in 1D edges: constant 0.0 (both models) → **drop**
> - `roughness` in 1D edges: constant 0.02 per model → drop from edge features (becomes model-level constant)
> - Event lengths vary: 94, 97, 205, 445 timesteps → must handle variable-length sequences
> - 2D roughness: only 3 unique values (0.013, 0.06, 0.10) → treat as-is, not encode

---

### 14.1 Target Transformation — The Most Critical Decision

Raw water level is a terrible prediction target because it encodes elevation (static baseline). The GNN should predict **physically meaningful residuals**.

**1D nodes:**
```python
# Water depth above the pipe invert (how full is the pipe?)
depth_above_invert = water_level - invert_elevation   # per-node normalization
# Range: [0, ∞) — physically can't go below invert
# Physical ceiling: surcharge when depth_above_invert > depth (pipe is pressurized)
```

**2D nodes:**
```python
# Inundation depth above the terrain minimum
water_depth = water_level - min_elevation   # per-node normalization  
# Range: [0, ∞) — physically can't go below terrain
```

**Why this matters:**
- M1 absolute 1D WL: std=16.87 ft. After subtracting invert_elevation: std ≈ 1–3 ft
- M2 absolute 1D WL: std=2.52 ft. After subtracting invert_elevation: std ≈ 0.5–1 ft
- Both models collapse to comparable scale → **enables shared backbone training**
- Physical constraints become simple lower-bound clipping at 0

**Delta prediction** (layer on top of depth transformation):
```python
# What the GNN actually predicts:
target = depth_above_invert[t] - depth_above_invert[t-1]   # 1D
target = water_depth[t] - water_depth[t-1]                  # 2D

# Reconstruct:
pred_wl_1d[t] = pred_wl_1d[t-1] + delta_1d + invert_elevation
pred_wl_2d[t] = pred_wl_2d[t-1] + delta_2d + min_elevation
```

---

### 14.2 Static Features: Keep, Drop, and Derive

#### 1D Node Static Features

| Feature | Action | Reason |
|---|---|---|
| `node_idx` | Drop (use as index) | Not a feature |
| `position_x` | Keep, **z-score per model** | Spatial position, but raw values are 800K+ → must normalize |
| `position_y` | Keep, **z-score per model** | Same |
| `depth` | Keep, **z-score per model** | Pipe capacity |
| `invert_elevation` | Use to **transform target only**; also keep normalized as feature | Corr=0.9997 with WL — the GNN needs this to understand elevation gradient |
| `surface_elevation` | Keep, **z-score per model** | Surcharge threshold |
| `base_area` | Keep, **z-score per model** | Storage capacity |

**Derived 1D node features:**
```python
# Surcharge risk: how close to flooding the surface?
fill_ratio = (surface_elevation - invert_elevation) / depth   # dimensionless, ~0 to 1

# Structural capacity proxy (Manning's full-pipe flow)
# Q_full ∝ D^(8/3) * slope^(1/2) / roughness
hydraulic_capacity_1d = (diameter ** (8/3)) * (slope ** 0.5)  # per edge → aggregate to node

# Connectivity (computed from edge_index)
degree_1d         = number of connected 1D edges per node
n_2d_connections  = number of 1D-2D coupling links per node  (from 1d2d_connections.csv)

# From 1D-2D connections: catchment area feeding into this node
connected_2d_area = sum(area of connected 2D nodes)  # from 2d_nodes_static
```

#### 2D Node Static Features

| Feature | Action | Reason |
|---|---|---|
| `node_idx` | Drop (use as index) | Not a feature |
| `position_x` | Keep, **z-score per model** | Spatial |
| `position_y` | Keep, **z-score per model** | Spatial |
| `area` | Keep, **log1p + z-score** | Skewed (cells vary in size) |
| `roughness` | Keep as-is (3 values) | Surface friction — already small scale |
| `min_elevation` | Use to **transform target only**; also keep normalized | Terrain floor |
| `elevation` | Keep **z-score per model** | Centroid elevation |
| `aspect` | Keep, **sin/cos encoding** | Directional → periodic feature; `sin(aspect_rad), cos(aspect_rad)` |
| `curvature` | Keep, **z-score per model** (clip outliers at ±3σ) | Water pooling/draining tendency |
| `flow_accumulation` | Keep, **log1p + z-score** | Only 16 unique values, but log scale is natural |

**Derived 2D node features:**
```python
# Water retention capacity before flooding starts
fill_capacity = (elevation - min_elevation) * area    # ft³ of deficit before inundation

# Terrain slope proxy (local gradient)
terrain_slope_proxy = (elevation - min_elevation) / np.sqrt(area)

# Upstream catchment volume proxy
catchment_potential = np.log1p(flow_accumulation) * area

# Coupling information
has_1d_connection    = binary (is this 2D node connected to a 1D drainage node?)
dist_to_nearest_1d   = Euclidean distance to nearest connected 1D node  # use position_x/y
connected_1d_depth   = depth of the coupled 1D node (drainage depth/capacity)
```

#### 1D Edge Static Features

| Feature | Action | Reason |
|---|---|---|
| `edge_idx` | Drop | Not a feature |
| `relative_position_x` | Keep, **z-score per model** | Flow direction info |
| `relative_position_y` | Keep, **z-score per model** | Flow direction info |
| `length` | Keep, **z-score per model** | Travel time |
| `diameter` | Keep, **z-score per model** | 4 unique values in M1 |
| `shape` | **DROP** | Constant 0.0 in both models |
| `roughness` | **DROP** | Constant 0.02 per model — adds nothing |
| `slope` | Keep, **z-score per model** | Critical for gravity-driven flow |

**Derived 1D edge features:**
```python
# Hydraulic gradient potential (slope × length = elevation drop)
head_loss = slope * length   # ft elevation drop across the pipe

# Manning's full-pipe capacity proxy (proportional to Q_full)
capacity_proxy = (diameter ** (8/3)) * (slope ** 0.5)  
# Note: roughness is constant → absorbed into model-level bias
```

#### 2D Edge Static Features

| Feature | Action | Reason |
|---|---|---|
| `edge_idx` | Drop | Not a feature |
| `relative_position_x` | Keep, **z-score per model** | Flow direction |
| `relative_position_y` | Keep, **z-score per model** | Flow direction |
| `face_length` | Keep, **z-score per model** | Opening width for flow |
| `length` | Keep, **z-score per model** | Cell centroid distance |
| `slope` | Keep, **z-score per model** | Gravitational driving force |

**Derived 2D edge features:**
```python
# Unit conveyance (flow potential per unit width)
unit_conveyance = face_length * (np.abs(slope) ** 0.5)   # Manning-like proxy

# Hydraulic head drop across edge
head_loss_2d = slope * length

# Flow direction (unit vector)
dx_norm = relative_position_x / (length + 1e-6)
dy_norm = relative_position_y / (length + 1e-6)
```

---

### 14.3 Dynamic Features at Each Timestep

**What is available at inference time (t ≥ 10)?**

| Feature | Test availability | Action |
|---|---|---|
| `rainfall[t]` | ✅ Always provided | Primary forcing input |
| `water_level[t-1]` | ✅ Predicted by GNN | Primary state variable |
| `water_volume[t-1]` | ❌ NaN post-warmup | **Do not use as GNN input** (unavailable at test time) |
| `inlet_flow[t-1]` | ❌ NaN post-warmup | **Do not use as GNN input** (unavailable at test time) |
| `edge flow/velocity[t-1]` | ❌ NaN post-warmup | **Do not use as GNN input** |

> **Rule**: The GNN's dynamic node state at t must be constructed using ONLY rainfall and the GNN's own previous prediction. No volume, no flow, no velocity as inputs post-warmup.

**Dynamic features to engineer (all derivable from rainfall + predicted WL):**

```python
# At timestep t for each node:

# 1. Current rainfall (always known)
x['rainfall_t'] = rainfall[t]    # 2D nodes only (0 for 1D nodes)

# 2. Cumulative rainfall (antecedent moisture proxy)
x['cumrain_6']  = sum(rainfall[t-5 : t+1])    # last 30 minutes
x['cumrain_12'] = sum(rainfall[t-11 : t+1])   # last 1 hour
x['cumrain_24'] = sum(rainfall[t-23 : t+1])   # last 2 hours
# These are the single most important dynamic features beyond current WL

# 3. Rainfall trend
x['rain_delta'] = rainfall[t] - rainfall[t-1]  # is it getting heavier or lighter?

# 4. Previous water states (from GNN predictions)
x['wl_prev_1']  = depth_transformed[t-1]   # depth above invert/min_elevation
x['wl_prev_2']  = depth_transformed[t-2]   # 2-step history
x['wl_delta_1'] = depth_transformed[t-1] - depth_transformed[t-2]  # velocity
x['wl_delta_2'] = depth_transformed[t-2] - depth_transformed[t-3]  # acceleration

# 5. Derived edge attribute at inference (from current predicted WL)
#    Hydraulic gradient (computed INSIDE GNN message passing, not pre-computed)
#    head_grad[i→j](t) = (wl[i](t-1) - wl[j](t-1)) / edge_length
#    This is the key physics signal that drives pipe/surface flow

# 6. Event position encoding
x['t_norm']  = t / total_event_length     # [0, 1] position in event
x['t_sin']   = sin(2π * t / event_length) # cyclical
x['t_cos']   = cos(2π * t / event_length) # cyclical
```

---

### 14.4 Graph Construction for HeteroGNN

**Edge types (4 types):**

```python
# Type 1: 1D pipe flow  (from 1d_edge_index.csv)
('1d_node', 'pipe_fwd', '1d_node')  # forward direction
('1d_node', 'pipe_rev', '1d_node')  # reverse (pipes are bidirectional — add reverse edges)

# Type 2: 2D surface flow  (from 2d_edge_index.csv)
('2d_node', 'surf_fwd', '2d_node')  # forward
('2d_node', 'surf_rev', '2d_node')  # reverse

# Type 3+4: 1D-2D coupling (from 1d2d_connections.csv)
('1d_node', 'to_surface', '2d_node')   # drainage node feeds surface
('2d_node', 'to_drain', '1d_node')     # surface water enters drain
```

**Graph scale (manageable in GPU memory):**
| | Model 1 | Model 2 |
|---|---|---|
| 1D nodes | 17 | 198 |
| 2D nodes | 3,716 | 4,299 |
| 1D edges × 2 (undirected) | 32 | 394 |
| 2D edges × 2 (undirected) | 15,870 | 19,752 |
| 1D-2D coupling links × 2 | 32 | 394 |
| **Total edges** | **~15,934** | **~20,540** |

This fits comfortably in 16 GB VRAM. One full event (445 timesteps × 4299 nodes × 128 dim) ≈ **900 MB** — process one event at a time.

---

### 14.5 Normalization Pipeline (Implementation Order)

```python
# Step 1: Compute statistics from TRAINING events only (no test leakage)
for model_id in [1, 2]:
    # 1D static features: compute mean/std from 1d_nodes_static.csv
    # 2D static features: compute mean/std from 2d_nodes_static.csv
    # 1D WL depth: collect all (water_level - invert_elevation) from all train events; compute std
    # 2D WL depth: collect all (water_level - min_elevation) from all train events; compute std
    # Rainfall: collect all values; compute 95th percentile for soft cap
    # Edge features: compute mean/std from static CSVs

# Step 2: Save normalization stats to JSON/pickle
# {model_id: {feature_name: {mean: float, std: float}, ...}}

# Step 3: Transform at data loading time (not pre-saved to disk — transform on-the-fly)
# Reason: pre-transforming 68 × 445 × 3716 rows is large; on-the-fly is memory-efficient
```

**Clipping strategy for robustness:**
```python
# Static features: clip continuous features at ±4σ after z-scoring (handle outliers in curvature)
# Dynamic depth: clip at 0 from below (physical floor), no upper clip during training
# During inference: clip predicted depth at 0 (physical constraint) before feeding back
```

---

### 14.6 Feature Summary Table

**1D node input vector at timestep t** (total ~20 dimensions):
```
[invert_elevation_norm, surface_elevation_norm, depth_norm, base_area_norm,   # 4 static
 fill_ratio, degree_1d_norm, n_2d_connections_norm, connected_2d_area_norm,   # 4 derived static
 position_x_norm, position_y_norm,                                            # 2 spatial
 wl_prev_1, wl_prev_2, wl_delta_1, wl_delta_2,                               # 4 temporal state
 t_norm, t_sin, t_cos]                                                        # 3 time encoding
```

**2D node input vector at timestep t** (total ~22 dimensions):
```
[elevation_norm, min_elevation_norm, area_log_norm, roughness,               # 4 static
 aspect_sin, aspect_cos, curvature_norm, flow_accum_log_norm,                # 4 static
 fill_capacity_norm, terrain_slope_proxy_norm, catchment_potential_norm,     # 3 derived static
 has_1d_connection, dist_to_1d_norm, connected_1d_depth_norm,               # 3 coupling info
 position_x_norm, position_y_norm,                                           # 2 spatial
 rainfall_t, cumrain_6_norm, cumrain_12_norm, rain_delta,                    # 4 rainfall
 wl_prev_1, wl_prev_2, wl_delta_1, wl_delta_2,                              # 4 temporal state
 t_norm, t_sin, t_cos]                                                       # 3 time encoding
```

**1D edge attribute vector** (total ~7 dimensions):
```
[rel_pos_x_norm, rel_pos_y_norm, length_norm, diameter_norm, slope_norm,     # 5 static
 head_loss_norm, capacity_proxy_norm]                                         # 2 derived
```

**2D edge attribute vector** (total ~7 dimensions):
```
[rel_pos_x_norm, rel_pos_y_norm, face_length_norm, length_norm, slope_norm,  # 5 static
 unit_conveyance_norm, dx_norm, dy_norm]                                      # wait → use 3 derived, total 8
```

---

### 14.7 Implementation Checklist

Execute in this order — each step validates the previous:

- [ ] **Step 1**: Load all static CSVs; compute + save normalization stats per model to `preprocessing_stats.json`
- [ ] **Step 2**: Build derived static features (fill_ratio, degree, coupling info, terrain proxies)
- [ ] **Step 3**: Validate: plot histograms of all features before/after normalization; check for any NaN or Inf
- [ ] **Step 4**: Build HeteroData graph for one event (M1/event_1) as a sanity check; use NetworkX to verify connectivity
- [ ] **Step 5**: Verify water depth transformation: confirm `depth_above_invert ≥ 0` for all train events; log any violations
- [ ] **Step 6**: Verify delta WL: confirm std ≈ 0.2 ft for M1, ≈ 0.18 ft for M2; this is what the GNN will predict
- [ ] **Step 7**: Profile memory: one full event as HeteroData tensor → measure VRAM usage
- [ ] **Step 8**: Implement `FloodEventDataset(torch.utils.data.Dataset)` that loads one event at a time with on-the-fly normalization
- [ ] **Step 9**: Run a sanity-check forward pass through a tiny GNN (2 layers, 32 dim) on one event; confirm output shapes
- [ ] **Step 10**: Confirm autoregressive inference loop: first 10 steps use GT, steps 11+ use predicted WL; verify physical clipping

---

### 14.8 Edge Case Handling

| Issue | Observed? | Fix |
|---|---|---|
| WL < invert_elevation (physical violation) | Possible near t=0 warmup | Clip `depth_above_invert = max(0, depth_above_invert)` |
| WL < min_elevation for 2D nodes | Possible | Clip `water_depth = max(0, water_depth)` |
| Rainfall = 0 for entire event (dry period) | Present in many events | No fix needed; model should learn 0 → no response |
| Variable event lengths (94 to 445 ts) | Confirmed | Handle per-event: no padding; GRU processes each event as its own sequence |
| M1 and M2 have wildly different WL scales | Confirmed (M1 ~300 ft, M2 ~40 ft) | Per-model normalization using depth transformation; shared backbone still works after this |
| 1D edges: shape + roughness are constant | Confirmed | Drop from edge features; the GNN gets no useful signal from these |
| `flow_accumulation` has only 16 unique values | Confirmed | Keep as continuous after log1p (effectively a coarse catchment class) |
| `curvature` may have outliers | Likely | Clip at ±4σ before normalizing |

