# UrbanFloodBench — Competition Analysis & Winning Strategy

## 1. Problem Summary

| Aspect | Detail |
|---|---|
| **Task** | Predict water levels at 1D (drainage) & 2D (surface) nodes across rainfall events |
| **Approach** | Autoregressive forecasting (optional — can use any method) |
| **Models** | 2 urban models (different layouts/hydraulics) |
| **Target** | `water_level` for every node at every timestep (after first 10 warmup steps) |
| **Metric** | Standardized RMSE (RMSE / std_dev per model×node_type) |
| **Submission** | Single CSV: `row_id, model_id, event_id, node_type, node_id, water_level` |

## 2. Metric Deep Dive — Standardized RMSE

The averaging hierarchy is **critical** for strategy:

```
node-level RMSE → avg over nodes → node_type score (1D, 2D separate)
→ avg(1D, 2D) → event score
→ avg over events → model score
→ avg(model_1, model_2) → FINAL SCORE
```

> [!IMPORTANT]
> **Key Implications:**
> - 1D and 2D nodes contribute **equally** despite vastly different node counts (2D >> 1D typically)
> - Each event contributes equally — poor performance on any single event hurts badly
> - Each model contributes equally — must perform well on BOTH models
> - RMSE is standardized by std_dev → errors are scale-invariant across models/types

## 3. Data Architecture

### Graph Structure
```mermaid
graph LR
    subgraph "Underground (1D)"
        A[1D Node] -->|1D Link/Pipe| B[1D Node]
        B -->|1D Link| C[1D Node]
    end
    subgraph "Surface (2D)"
        D[2D Node] -->|2D Link| E[2D Node]
        E -->|2D Link| F[2D Node]
    end
    A -.->|1D-2D Coupling| D
    C -.->|1D-2D Coupling| F
```

### Feature Summary

| Category | Features | Key Variables |
|---|---|---|
| **1D Static Nodes** | position_x/y, depth, invert_elev, surface_elev, base_area | 6 features |
| **1D Static Edges** | length, rel_pos_x/y, diameter, shape, roughness, slope | 7 features |
| **1D Dynamic Nodes** | water_level (TARGET), inlet_flow | 2 features |
| **1D Dynamic Edges** | flow, velocity | 2 features |
| **2D Static Nodes** | position_x/y, area, roughness, min_elev, centroid_elev, aspect, curvature, flow_accum | 9 features |
| **2D Static Edges** | rel_pos_x/y, face_length, 2d_length, slope | 5 features |
| **2D Dynamic Nodes** | rainfall (INPUT), water_level (TARGET), water_volume | 3 features |
| **2D Dynamic Edges** | flow, velocity | 2 features |

### Test Set Structure
- **First 10 timesteps**: All dynamic features provided (warmup)
- **After timestep 10**: Only **rainfall** is provided as input
- Participants must predict `water_level` for timesteps 11+ at all nodes

## 4. Strategic Observations

### 4.1 This is a Graph Problem
- Nodes + edges + connectivity = **Graph Neural Network (GNN)** is the natural fit
- The 1D-2D coupling links make this a **heterogeneous graph**
- Message passing can propagate flood signals through the network

### 4.2 Spatial + Temporal
- Temporal: water level at t depends on t-1, t-2, etc.
- Spatial: water level at node i depends on upstream/neighboring nodes
- → Need **spatio-temporal architecture** (GNN + temporal model)

### 4.3 Physics Priors
- Rainfall → surface accumulation → drainage entry → pipe flow → downstream transport
- Conservation of mass/volume at each node
- Elevation drives gravity-based flow
- Manning's equation relates flow to roughness, slope, hydraulic radius

### 4.4 Autoregressive Risk
- Error accumulation over time is the biggest challenge
- Teacher forcing during training, scheduled sampling for robustness
- Multi-step loss or curriculum learning to mitigate drift

## 5. Winning Strategy — Multi-Tier Approach

### Tier 1: Strong Baseline (Days 1-3)
**Architecture: Per-node temporal model with spatial features**
- For each node, create feature vectors from its static features + neighbor aggregations
- Use **LightGBM/XGBoost** for water_level(t+1) prediction given recent history
- Feature engineering: rolling stats, delta features, lag features
- Train separate models for 1D and 2D nodes

### Tier 2: Graph Neural Network (Days 3-7)
**Architecture: Spatio-Temporal GNN**
- **Encoder**: Embed static features for nodes and edges
- **Message Passing**: Use heterogeneous GNN (GAT or GraphSAGE) with separate edge types for 1D, 2D, and 1D-2D links
- **Temporal**: GRU/LSTM cells at each node, or Transformer-style temporal attention
- **Decoder**: Predict water_level at next timestep
- **Training**: Teacher forcing + scheduled sampling + multi-step loss

### Tier 3: Advanced Techniques (Days 7-14)
- **Physics-Informed Loss**: Add soft constraints (mass conservation, non-negative depth)
- **Multi-task Learning**: Jointly predict water_volume, flow, velocity as auxiliary targets
- **Ensemble**: Combine GNN + gradient boosting + potentially a simple MLP baseline
- **Per-Model Specialization**: Train separate models for Model 1 and Model 2
- **Data Augmentation**: Scale rainfall intensities, temporal jittering
- **Test-Time Adaptation**: Use the 10 warmup steps to fine-tune/calibrate

### Tier 4: Competition Edge
- **Curriculum Learning**: Train on easy events first, then hard ones
- **Multi-horizon Loss**: Weight near-future predictions higher
- **Attention over Events**: Cross-event training to learn generalizable dynamics
- **Post-processing**: Clip predictions to physical bounds (e.g., water level ≥ invert elevation for 1D)

## 6. Recommended Model Architecture

```
┌─────────────────────────────────────────────┐
│              INPUT AT TIME t                │
│  Static features + rainfall(t) + state(t)   │
├─────────────────────────────────────────────┤
│         NODE/EDGE EMBEDDING LAYER           │
│  Separate encoders for 1D/2D nodes & edges  │
├─────────────────────────────────────────────┤
│      HETEROGENEOUS GNN (2-3 layers)         │
│  Message passing: 1D↔1D, 2D↔2D, 1D↔2D      │
├─────────────────────────────────────────────┤
│          TEMPORAL MODULE (GRU)              │
│  Per-node hidden state + temporal context   │
├─────────────────────────────────────────────┤
│           PREDICTION HEAD                   │
│  water_level(t+1) for all nodes             │
└─────────────────────────────────────────────┘
           ↓ (autoregressive loop)
     Feed predictions back as state(t+1)
```

## 7. Next Steps

1. **EDA Notebook**: Explore data shapes, distributions, correlations, and graph structure
2. **Baseline**: Build a quick per-node regression baseline to understand difficulty
3. **GNN Implementation**: Build the core spatio-temporal GNN
4. **Iterate**: Train, evaluate, tune, ensemble
