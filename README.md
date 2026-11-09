# 📡 Telecommunications Network Digital Twin

A Python-based telecommunications network simulation that models a fictional town-wide 4G/5G environment. The project explores mobile-user behaviour, radio access network (RAN) performance, cell-site capacity, network faults and service quality through an interactive Streamlit dashboard.

The aim is to create a simplified digital representation of a mobile network that can be used to explore network behaviour, monitor performance and simulate operational scenarios.

---

## Dashboard Preview
The interactive dashboard provides a live view of the simulated
telecommunications network, including user locations, cell sites
and network performance.

![Telecommunications Digital Twin Dashboard](Screenshots/cover.png)
## Project Overview

This project simulates a telecommunications network across a configurable geographical area containing multiple regions, mobile users and cellular towers.

Users are assigned different profiles, devices, network capabilities and activities. Their movement through the simulated environment affects their connection to cell sites and the radio conditions they experience.

The network continuously calculates performance metrics and responds to events such as tower outages, poor signal quality, handovers and network congestion.

An interactive dashboard provides a visual representation of the simulated network, allowing users to monitor performance, investigate individual devices and introduce network fault scenarios.

**This is an educational and experimental model. It is not a standards-compliant LTE or 5G network simulator.**

## Key Features

### 1. Synthetic Town and Network Infrastructure

- Generates a configurable geographical environment using a spatial grid.
- Divides the town into regions with different population densities.
- Classifies regions as urban, residential, suburban or rural.
- Estimates regional population.
- Allocates cellular towers proportionally to regional population.
- Places individual towers across the simulated environment.

### 2. Mobile User Simulation

- Generates configurable numbers of simulated mobile users.
- Assigns user profiles, including residents, commuters and visitors.
- Models different device types and network capabilities.
- Assigns user activities such as browsing, streaming, voice calls and uploading.
- Generates movement profiles for stationary, walking and driving users.
- Updates user positions and destinations over simulation time.

### 3. Radio Access Network (RAN) Modelling

Models simplified radio conditions between mobile users and cell sites.

Implemented calculations include:

- Distance between users and serving towers.
- Radio path loss.
- Shadow fading.
- Received signal power.
- Co-channel interference.
- Thermal noise.
- Signal-to-Interference-plus-Noise Ratio (SINR).

These calculations provide the basis for evaluating the quality of individual network connections.

### 4. Network Connections and Handovers

- Assigns users to their nearest operational tower during initialisation.
- Identifies neighbouring operational towers.
- Evaluates alternative serving-cell radio conditions.
- Implements handover decisions using signal-strength hysteresis and a time-to-trigger mechanism.
- Tracks successful and failed handovers.
- Detects poor signal conditions and connection drops.
- Attempts automatic reconnection to suitable operational towers.

### 5. Network Performance and Capacity

Models the relationship between radio conditions, user demand and available cell capacity.

Performance calculations include:

- Achievable data rate using the Shannon capacity formula.
- User-level throughput.
- Cell-level capacity and resource allocation.
- Traffic demand by user activity.
- Cell capacity utilisation.
- Network congestion classification.
- Queueing delay.
- Network latency.
- Packet loss.

Cell resources are shared between connected users, with allocations constrained by estimated radio capacity and traffic demand.

### 6. Network Faults and Recovery

Introduces operational scenarios to explore how network performance changes when infrastructure or connections fail.

Features include:

- Probabilistic cell-site outages.
- Automatic tower recovery.
- Detection of users affected by unavailable towers.
- Connection-drop tracking.
- Automatic user reconnection.
- Handover and recovery counters.
- Manual tower outage and restoration through the dashboard.

### 7. Interactive Streamlit Dashboard

The dashboard provides a live visual interface for monitoring and interacting with the simulation.

**Network overview**

- Interactive geographical network map.
- User and tower locations.
- Coverage circles.
- Connection states.
- Visualisation of pending handovers.
- Network performance trends.

**RAN and cell capacity**

- Tower-level performance metrics.
- Connected-user counts.
- Cell capacity and utilisation.
- Throughput and congestion information.

**User explorer**

- Individual user inspection.
- Serving tower information.
- Signal quality and throughput.
- Latency and packet loss.
- User movement and connection status.

**Faults and reports**

- Tower operational status.
- Outage and recovery counters.
- Simulation event log.
- Manual fault injection.
- High-traffic scenario controls.

**Data exports**

- Filtered user records in CSV format.
- Tower status in CSV format.
- Excel performance reports containing user, cell, tower, region and network-summary information.

---

## Technologies Used

| Technology | Purpose                                      |
| ---------- | -------------------------------------------- |
| Python     | Simulation engine and application logic      |
| NumPy      | Numerical calculations and spatial modelling |
| Pandas     | Data manipulation and performance analysis   |
| Streamlit  | Interactive dashboard                        |
| Plotly     | Interactive charts and network visualisation |
| Matplotlib | Plotting support                             |

SQLite is being explored for persistent storage of simulation configurations and historical network snapshots.

---

## Project Structure

```text
telecommunications-network-digital-twin/
│
├── main.py
│   └── Simulation controller and Streamlit dashboard
│
├── town.py
│   └── Town generation, population and user mobility
│
├── sites.py
│   └── RAN modelling and network performance calculation
│
├── README.md
│
└── .gitignore
```

---

## Installation and Setup

### Prerequisites

- Python 3.10 or later.
- Git.
- A code editor such as Visual Studio Code.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/telecommunications-network-digital-twin.git
```

### 2. Navigate into the project

```bash
cd telecommunications-network-digital-twin
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install numpy, pandas, streamlit, plotly, matplotlib
```

### 5. Run the dashboard

```bash
streamlit run telecommunications_dashboard.py
```

Streamlit will provide a local URL through which the dashboard can be accessed in a web browser.

---

## Simulation Configuration

The dashboard provides configurable simulation parameters, including:

| Parameter            | Description                        |
| -------------------- | ---------------------------------- |
| Number of users      | Number of simulated mobile devices |
| Number of towers     | Number of cellular sites           |
| Number of regions    | Geographical regions in the town   |
| Grid resolution      | Spatial resolution in metres       |
| Coverage range       | Estimated cell coverage radius     |
| Time step            | Simulated seconds per update       |
| Outage probability   | Probability of a tower outage      |
| Recovery probability | Probability of tower recovery      |

The simulation can be initialised, started, paused or advanced one step at a time.

---

## Performance Metrics

The simulation calculates a range of metrics to assess network behaviour.

| Metric               | Description                                       |
| -------------------- | ------------------------------------------------- |
| SINR                 | Signal quality relative to interference and noise |
| Received Power       | Signal power received by a mobile device          |
| Throughput           | Estimated data delivery rate                      |
| Latency              | Estimated network delay                           |
| Packet Loss          | Estimated percentage of lost packets              |
| Capacity Utilisation | Proportion of estimated cell capacity being used  |
| Connection Rate      | Percentage of users currently connected           |
| Handover Count       | Number of completed serving-cell changes          |
| Dropped Sessions     | Number of dropped user sessions                   |
| Tower Outages        | Number of towers currently unavailable            |

---

## Current Development

The project is being developed in stages, with an emphasis on understanding the underlying telecommunications and software concepts.

| Component                             | Development status |
| ------------------------------------- | ------------------ |
| Synthetic town generation             | Implemented        |
| Population modelling                  | Implemented        |
| User generation and mobility          | Implemented        |
| Tower placement and association       | Implemented        |
| Radio propagation and SINR modelling  | Implemented        |
| Handovers and connection management   | Implemented        |
| Cell capacity and resource allocation | Implemented        |
| Congestion and network performance    | Implemented        |
| Tower outage and recovery scenarios   | Implemented        |
| Interactive Streamlit dashboard       | Implemented        |
| CSV and Excel reporting               | Implemented        |
| SQLite database persistence           | In development     |
| Historical simulation analysis        | Planned            |
| Machine learning extensions           | Planned            |

---

## Future Improvements

Potential future extensions include:

- Completing SQLite integration to store simulation configurations and historical user, tower and network snapshots.
- Building SQL queries to analyse and compare separate simulation runs.
- Improving model calibration and validating assumptions against published telecommunications models.
- Introducing machine learning to investigate network performance patterns and predict potential faults or congestion.
- Expanding network scenarios and testing different infrastructure configurations.

---

## Project Objectives and Learning Outcomes

This project has provided practical experience in:

- Developing a multi-component Python application.
- Working with NumPy and Pandas for numerical modelling and data manipulation.
- Designing and implementing a simplified radio access network simulation.
- Modelling mobile-user behaviour and network mobility.
- Understanding signal quality, interference, SINR and throughput.
- Implementing network fault scenarios and automated recovery mechanisms.
- Building interactive dashboards using Streamlit and Plotly.
- Organising code into reusable functions and separate modules.
- Investigating software issues through testing and debugging.
- Exploring SQL database design and data persistence.

The wider objective is to develop a practical understanding of telecommunications network behaviour, software development, technical troubleshooting and network performance analysis.

---

## Disclaimer

This project is a fictional, simplified telecommunications simulation developed for educational and portfolio purposes.

It does not represent a real mobile operator's network and should not be used for live network planning or engineering decisions. Radio propagation, interference, cell capacity, handover and performance calculations use simplified assumptions rather than full 3GPP-compliant models. The Author takes no responsibility for any results or use coming from the simulation. Always consult a professional.

---

**Author:** Christopher Grossmann

**Project:** Telecommunications Network Digital Twin

**Language:** Python
