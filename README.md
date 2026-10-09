# Developing Your First ML Workflow

Companion code for the Udacity course "Developing Your First ML Workflow" (part of the Machine Learning DevOps Engineer program). The repo holds the demo notebooks, exercise notebooks with worked solutions, and the final project starter used throughout the course. Everything runs on Amazon SageMaker and supporting AWS services.

## What you learn

* How to build an end to end ML workflow on AWS: preprocess data, train a model, deploy it, and monitor it in production.
* How to use SageMaker training jobs, endpoints, batch transform, and processing jobs.
* How to write AWS Lambda functions and chain them into automated pipelines with Step Functions.
* How to use SageMaker Feature Store, Model Monitor, and Clarify for feature management, production monitoring, and model explainability.

## Folder structure

* `lesson2/` - SageMaker essentials. Exercises 1 to 5 cover training jobs, endpoints, batch transform, and processing jobs, plus a full XGBoost sentiment analysis walkthrough. Each exercise ships with a worked solution notebook, and `lesson2/demo/` holds the short lecture demos.
* `lesson3/` - Designing workflows. Exercises cover Lambda functions, invoking Lambda synchronously and asynchronously, Step Functions, and tying it all together. Includes `HelloBlazePreprocess.py` (SageMaker processing entry point) and `HelloBlazePreprocessLambda.py` (Lambda version of the same pipeline).
* `lesson4/` - Feature Store, Model Monitor, and Clarify. `demos.ipynb` walks through each service, `exercises-starters.ipynb` has the student exercises, and `exercises-solutions.ipynb` has the worked solutions.
* `project/` - Final project starter: build, deploy, and monitor an image classification workflow (bicycles vs motorcycles from CIFAR-100) with a SageMaker endpoint, Lambda functions, Step Functions, and Model Monitor.

## How to run

1. Install dependencies:
   `pip install -r requirements.txt`
2. Open the notebooks in Amazon SageMaker Studio or a SageMaker notebook instance (the course was tested on the Python 3 Data Science kernel). A local Jupyter install works for reading, but most cells call AWS services.
3. Configure AWS credentials with permissions for SageMaker, S3, Lambda, and Step Functions, and fill in the bucket names marked `CHANGE THIS` / `FILL_THIS_IN` in the exercise notebooks and scripts.
4. Run the cells top to bottom. Exercise notebooks contain `TODO` sections for you to complete; the `* Solution.ipynb` notebooks show one worked way to complete them.

Notes:

* Running these notebooks launches real AWS resources (training jobs, endpoints, processing jobs). Endpoints and monitoring schedules cost money while they run, so delete them when you finish an exercise. Several notebooks include cleanup cells for this.
* The large `.zip` data files in `lesson2/` and `lesson3/` are the Amazon review datasets used by the exercises; they are needed for the preprocessing cells.
* Some cells reference example S3 paths and IAM role ARNs from the original course account. Replace them with your own bucket and role before running.

## License

See `LICENSE.md`.
