# UAV Intrusion Detection Using Federated Continuous Learning

## Overview

This project demonstrates a federated learning approach to detect anomalies and intrusions in Unmanned Aerial Vehicles (UAVs), particularly in swarm-based operations. It leverages a CNN-LSTM model within a federated learning framework to improve detection accuracy and efficiency across heterogeneous UAV datasets while preserving data privacy.

## Abstract

Unmanned Aerial Vehicles (UAVs) are getting wide acceptance from different sectors, including public services, military, emergency response, and commercial applications. While the potential benefits of UAVs are growing significantly, they can exhibit unexpected behavior due to sensor malfunction, unforeseen environmental circumstances, or power outages. These anomalies can severely affect UAV missions, especially in swarm-based UAV operations, by compromising decision-making and trajectory planning processes. Traditional intrusion detection systems (IDS) typically rely on binary classification for individual UAVs using computationally expensive neural networks, limiting their ability to predict multiple types of attacks while operating within the resource-constrained environments of real-world UAV scenarios. To address these challenges, we propose a federated continuous learning approach to facilitating decentralized training across diverse UAV swarms using heterogeneous datasets while preserving data privacy. By leveraging a lightweight CNN-LSTM model that captures both spatial and temporal features, our method significantly improves multi-class classification accuracy while using a model that has five times fewer parameters and computationally faster than traditional approaches. This research showcases the potential of federated learning to enhance the security of UAV swarm networks through robust multi-class classification.

### Detection Accuracies:
- **TLM-UAV Dataset:** 96.85%
- **UKM-IDS Dataset:** 99.45%
- **UAV-IDS Dataset:** 99.99%
- **Cyber-Physical Dataset:** 98.05%

## Directory Structure

```plaintext
project_root/
├── clients/
│   ├── client_cyber_physical.py
│   ├── client_TLM_UAV.py
│   ├── client_UAV_IDS.py
│   ├── client_UKM_IDS.py
├── preprocessing/
│   ├── preprocess_cyber_physical.py
│   ├── preprocess_TLM_UAV.py
│   ├── preprocess_uav_ids.py
│   ├── preprocess_ukm_ids.py
├── models/
│   ├── encoder_classifier_cnn_lstm.py
├── server/
│   ├── server.py
├── utils/
│   ├── metrics.py
├── data/
│   ├── cyber_physical/
│   │   ├── train.csv
│   │   ├── test.csv
│   ├── tlm_uav/
│   │   ├── train.csv
│   │   ├── test.csv
│   ├── uav_ids/
│   │   ├── train.csv
│   │   ├── test.csv
│   ├── ukm_ids/
│   │   ├── train.csv
│   │   ├── test.csv
├── result/
│   ├── client_cyber_physical/
│   ├── client_TLM_IDS/
│   ├── client_UAV_IDS/
│   ├── client_UKM_IDS/
├── README.md

```

## Requirements

- **CUDA Toolkit 12.6** (for GPU acceleration)
- **Python 3.12.5**


## Setup Instructions

### Clone the Repository

```bash
git clone https://github.com/Kanchon-Gharami/UAV-IDS-FL
cd UAV-IDS-FL
```


### Create and Active Virtual Environment
```bash
python3 -m venv venv
venv\Scripts\activate

```

### Install Required Packages
```bash
pip install -r requirements.txt
```
_Ensure CUDA 12.6-compatible versions of dependencies are installed for GPU support._


## Running the Code

### Start the Server
In a separate terminal, start the federated learning server:
```bash
python server/server.py
```

### Start the Clients
In separate terminals, run the clients to start training on each dataset.
```bash
python clients/client_cyber_physical.py
python clients/client_TLM_UAV.py
python clients/client_UAV_IDS.py
python clients/client_UKM_IDS.py

```
### Results
Our approach achieved high detection accuracy across multiple datasets, showcasing the effectiveness of federated learning for UAV swarm network security. The results can be viewed in the ```result/``` directory for each client.

## 📖 Citation

If you use this repository for your research, please cite our paper accepted at the **44th AIAA/IEEE Digital Avionics Systems Conference (DASC)**:

*Kanchon Gharami* and *Shafika Showkat Moni*, "An Efficient Privacy-preserving Intrusion Detection Scheme for UAV Swarm Networks",
44th AIAA/IEEE Digital Avionics Systems Conference (DASC), 2025. Accepted, in publication.

**BibTeX:**
```bibtex
@inproceedings{gharam2025dasc,
  author    = {Kanchon Gharami and Shafika Showkat Moni},
  title     = {An Efficient Privacy-preserving Intrusion Detection Scheme for UAV Swarm Networks},
  booktitle = {Proceedings of the 44th AIAA/IEEE Digital Avionics Systems Conference (DASC)},
  year      = {2025},
  note      = {Accepted, in publication},
  url       = {https://github.com/Kanchon-Gharami/UAV-Trajectory-Prediction}
}
```


## Contact
For questions or issues, please contact gharamik@my.erau.edu



