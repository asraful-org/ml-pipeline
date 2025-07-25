
<h2>ADR-001 : ML Pipeline Design & Architecture Decisions</h2>

<h3> 📅Date: 2025-07-17 </h3>
<h3> 📌 Status: Approved </h3>
<h3> 👥 Deciders: ML/Ops Engineering Team</h3>


## Context

We need to simulate the deployment of a machine learning model to production, starting from a greenfield project through to a proof of concept (POC).

---
## Decision Drivers

- **Team Size:** Solo team
- **Complexity:** Keep it simple, avoid over-engineering
- **Extendibility:** Choose familiar, well-documented tools
- **Timeline:** Need production deployment in 3 days
- **Budget:** Accommodate with Azure Free Tier

## Decisions
- Orchestration Strategy: GitHub Actions
- Chosen: GitHub Actions for CI/CD
- Combined training pipeline for training and deployment 

## Rationale:
✅ Zero additional infrastructure cost
✅ Team already familiar with GitHub workflows
✅ Sufficient for current POC

**Rejected Alternatives:**

- **Airflow:** Too complex for our current needs, requires separate infrastructure

## Implementation Plan
Day 1/2/3: Setup GitHub Actions workflow


## Consequences

##Positive

- Low operational overhead
- Fast time to production
- Familiar tools for team
- Cost-effective solution

## Negative

- Limited to GitHub's compute resources
- Manual scaling required as complexity grows
- Less sophisticated scheduling compared to dedicated orchestrators