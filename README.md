# FlexTrack Challenge 2025 - Overview (GPT generated)

This document summarizes the context, goals, and ecosystem relevant to the [FlexTrack Challenge 2025](https://www.aicrowd.com/challenges/flextrack-challenge-2025), focusing on building power consumption, demand response (DR), and the Swiss power system.

## 1. Swiss Power System Overview

### Electricity Generation

* Produced by hydro, nuclear, thermal, wind, solar, and other sources.
* Feeds into the **transmission grid** managed by Swissgrid.

### Transmission Grid (Swissgrid)

* Operates the high-voltage backbone (380/220 kV).
* Moves electricity across regions and balances supply & demand in real time.

### Distribution Grids (DSOs)

* Operated by companies such as **CKW**, **BKW**, and **Axpo Grid**.
* Deliver electricity to buildings and homes.
* Maintain infrastructure and implement smart grid solutions.

### Consumers (Buildings)

* Consume electricity via HVAC, lighting, equipment, etc.
* Usage patterns vary by day, season, and occupancy.

## 2. Demand Response (DR)

### Concept

* Buildings flex their electricity use to help balance the grid:

  * **Reduce load** during peaks (shed energy).
  * **Increase load** when excess generation is available (absorb energy).
* DR events are tracked via a **flag** (−1, 0, +1) and **capacity** (kW change).

### Participants

* **Grid Operators**: Swissgrid and DSOs coordinate DR.
* **Building Managers / Aggregators**: Implement DR and report flexibility.

## 3. Interaction Between Companies

| Actor                       | Role in DR / Power Flow                                           |
| --------------------------- | ----------------------------------------------------------------- |
| Swissgrid                   | Manages transmission grid; ensures large-scale balance            |
| DSOs (CKW, BKW, Axpo, etc.) | Deliver electricity; implement local DR; collect consumption data |
| Buildings / Consumers       | Execute DR events; provide flexibility                            |
| Aggregators                 | Pool multiple buildings' flexibility and offer to grid operators  |

**Flow:** `Generation → Transmission (Swissgrid) → Distribution (DSOs) → Buildings`
*DR signals* flow from grid operators/aggregators to buildings, while consumption and flexibility data flow back upstream.

## 4. FlexTrack Challenge 2025 Goals

* **Objective:** Build AI models to detect and measure building flexibility.
* **Tasks:**

  1. **Classification:** Detect DR events (when buildings are flexing).
  2. **Regression:** Predict DR capacity (how much buildings changed consumption).
* **Impact:**

  * Enables grid operators to balance supply & demand efficiently.
  * Helps building operators validate and optimize their DR contributions.
  * Supports aggregators in reliable DR aggregation.
  * Facilitates integration of renewable energy.

**Outcome:** Improved measurement and prediction of building flexibility, leading to smarter, more stable, and renewable-friendly grid operation.

