# Productionizing a simple ML model

# 📋 Project Requirements

### Issue: [#1 Requirements](https://github.com/asraful-org/ml-pipeline/issues/21)

---


##  Task Status

``` 
 Total process is automated through Github Actions workflow 
 Extensive monitoring is achieved by utilizing Azure native monitoring capability 
```


- ### Pipeline Design (Basic CI/CD + Reproducibility)
    - *Achieved with Github Actions and Azure ML*
- ### Containerized Inference API
    - *Achieved with Docker, Azure Container Registry , Azure Container Isnt, Azure APIM*
- ### Basic Monitoring
    - *Achieved with Azure ML Dashboard and Azure ML logging*


# How to Run the Training Pipeline
- The training pipeline, defined in the train_model job of the CI/CD workflow, is responsible for ingesting data, training the machine learning model, and saving the trained artifacts.

## Prerequisites:

- This **pipeline runs automatically on push to the main branch** 

- Azure credentials (Client ID, Tenant ID, Subscription ID) must be configured as GitHub Secrets for OIDC authentication.

- The requirements.txt file must specify all necessary Python libraries for training (e.g., azure-ai-ml, azure-identity, mlflow, and your model's dependencies like scikit-learn).

## Monitoring Training Progress
 - GitHub Actions Logs
 - MLflow in Azure ML Workspace


# How to Build and Run the Inference Container

- The inference container encapsulates the trained model and its dependencies into a Docker image, which is then deployed to Azure Container Instances (ACI). This process is handled by the build_push_docker_image and deploy_to_aci jobs.

# What is Monitored and How
 - Training Pipeline Monitoring
 - Inference Container Monitoring
 - API activity 
 - Model monitoring : Azure ML 



## 📄 Documentation/Approach
- [📋 ADR-001 : ML Pipeline Design & Architecture Decisions](ADR-001_ML_Pipeline_Design_&_Architecture_Decisions.md)
- [📋 ADR-002 : Azure ML Platform Architecture](ADR-002_Azure_ML_Platform_Architecture.md)


## 📄 Azure Infrastructure / Manual Provisioning  

- Based on [📋 ADR-002 ](ADR-002_Azure_ML_Platform_Architecture.md) i have provisioned azure resources manually for this POC

## Trade-off 
 - I have chosen a simple/MVP style pipeline over doing everything in POC


## 📄 Next Steps / Improvement  

- Asses Azure Devops and compare it with current github flow (integration could be easy)
- Narrow down role based permission at Azure service principle  (Azure APP Register)
- Introduce secure virtual network
- Introduce infrastructure as a code (IoC) using terraform 
- Provision Azure infrastructure automatically (CI/CD)
- Introduce proper monitoring and observability 



## Acknowledgement of Generative AI Assistance

I acknowledge the partial use of generative AI tools. Specifically:

-  Generative AI was used to help with **boilerplate generation** (code, scripts, documents)



## 📄 References

- [Architecture best practices for Azure Machine Learning](https://learn.microsoft.com/en-us/azure/well-architected/service-guides/azure-machine-learning)

