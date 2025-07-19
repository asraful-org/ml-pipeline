
<h2>ADR-002: Azure ML Platform Architecture</h2>

<h3> 📅 Date: 2025-07-17 </h3>  
<h3> 📌 Status: Approved  </h3>
<h3> 👥 Deciders: ML/Ops Engineering/Infra Team  </h3>



## Context

We ran a quick POC using local ML FLow installation and exposing it using ngrok to enable GithubActions to register model to remote ML Flow tracking server.
Now, we need a simulation with Azure infrastructure. 

- **Compliance:** GDPR data governance requirements
- **Scale:** Multiple models, teams, and environments
- **Security:** Enterprise-grade authentication and network isolation/ possible
- **Collaboration:** Cross-team model sharing and governance
- **SLA Requirements:** 99.9% uptime for inference services

---

## Decision Overview

## Platform

**Azure ML ecosystem with integrated services**

## Component Overview

| Component           | Service                         | Purpose                                |
|---------------------|----------------------------------|----------------------------------------|
| Experiment Tracking | Azure ML Studio                 | Replace self-hosted MLflow             |
| Model Registry      | Azure ML Model Registry         | Centralized model versioning           |
| Container Registry  | Azure Container Registry (ACR)  | Secure image storage                   |
| Model Serving       | Azure Container Instances (ACI) | Scalable inference deployment          |
| API Gateway         | Azure API Management            | Enterprise API governance              |

