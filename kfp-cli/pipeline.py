import google.cloud.aiplatform as aiplatform
import kfp
from kfp import compiler, dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output, component, ClassificationMetrics
import os 
from collections import namedtuple
from typing import NamedTuple


PROJECT_ID = os.environ['PROJECT_ID'] # replace with project ID
REGION = os.environ['REGION']
EXPERIMENT = 'vertex-pipelines'

# gcs bucket
GCS_BUCKET = PROJECT_ID
BUCKET_URI = f"gs://{PROJECT_ID}-bucket"  # @param {type:"string"}
PIPELINE_ROOT = PIPELINE_ROOT = BUCKET_URI + "/pipeline_root/"

# BUCKET_NAME="gs://" + PROJECT_ID + "-bucket"
PIPELINE_ROOT = BUCKET_URI + "/pipeline_root/"
aiplatform.init(project=PROJECT_ID, staging_bucket=BUCKET_URI)

@dsl.component(base_image='python:3.8', 
    packages_to_install=[
        "pandas==1.3.4",
        "scikit-learn==1.0.1",
        "google-cloud-bigquery==3.13.0",
        "db-dtypes==1.1.1"
    ],
)
def get_data(
    project_id: str,
    dataset_train: Output[Dataset],
    dataset_test: Output[Dataset],
) -> None:
    
    """ Loads data from BigQuery, splits it into training and test sets,
    and saves them as CSV files.

    Args:
        project_id: str
        dataset_train: Output[Dataset] for the training set.
        dataset_test: Output[Dataset] for the test set.
    """

    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import pandas as pd

    from google.cloud import bigquery

    # Construct a BigQuery client object.
    client = bigquery.Client(project= project_id)
    job_config = bigquery.QueryJobConfig()
    query = """

        SELECT
      * EXCEPT(fullVisitorId)
    FROM

      # features
      (SELECT
        fullVisitorId,
        IFNULL(totals.bounces, 0) AS bounces,
        IFNULL(totals.timeOnSite, 0) AS time_on_site
      FROM
        `data-to-insights.ecommerce.web_analytics`
      WHERE
        totals.newVisits = 1
        AND date BETWEEN '20160801' AND '20170430') # train on first 9 months
      JOIN
      (SELECT
        fullvisitorid,
        IF(COUNTIF(totals.transactions > 0 AND totals.newVisits IS NULL) > 0, 1, 0) AS will_buy_on_return_visit
      FROM
          `data-to-insights.ecommerce.web_analytics`
      GROUP BY fullvisitorid)
      USING (fullVisitorId)
      LIMIT 10000
    ;
    """

    query_job = client.query(query, job_config=job_config)
    df = query_job.to_dataframe()
    
    # Split Data
    train, test = train_test_split(df, test_size=0.3, random_state=42)

    # Save to Outputs
    train.to_csv(dataset_train.path, index=False)
    test.to_csv(dataset_test.path, index=False)
    
    
    

@dsl.component(base_image='python:3.8', 
    packages_to_install=[
        "xgboost==1.6.2",
        "pandas==1.3.5",
        "joblib==1.1.0",        
        "scikit-learn==1.0.2",
    ],
)
def train_model(
    dataset: Input[Dataset],
    model_artifact: Output[Model], 
) -> None:

    """Trains an XGBoost classifier on a given dataset and saves the model artifact.

    Args:
        dataset: Input[Dataset]
            The training dataset as a Kubeflow component input.
        model_artifact: Output[Model]
            A Kubeflow component output for saving the trained model.

    Returns:
        None
            This function doesn't have a return value; its primary purpose is to produce a model artifact.
    """
    import os
    import joblib
    import pandas as pd
    from xgboost import XGBClassifier

    # Load Training Data
    data = pd.read_csv(dataset.path)

    # Train XGBoost Model
    model = XGBClassifier(objective="binary:logistic")
    model.fit(data.drop(columns=["will_buy_on_return_visit"]), data.will_buy_on_return_visit)

    # Evaluate and Log Metrics
    score = model.score(data.drop(columns=["will_buy_on_return_visit"]), data.will_buy_on_return_visit)

    # Save the Model Artifact
    os.makedirs(model_artifact.path, exist_ok=True)
    joblib.dump(model, os.path.join(model_artifact.path, "model.joblib"))

    # Metadata for the Artifact
    model_artifact.metadata["train_score"] = float(score)
    model_artifact.metadata["framework"] = "XGBoost"


    
@dsl.component(base_image='python:3.8', 
    packages_to_install=[
        "xgboost==1.6.2",
        "pandas==1.3.5",
        "joblib==1.1.0",
        "scikit-learn==1.0.2",
        "google-cloud-storage==2.13.0",
    ],
)
def eval_model(
    test_set: Input[Dataset],
    xgb_model: Input[Model],
    metrics: Output[ClassificationMetrics],
    smetrics: Output[Metrics],
    bucket_name: str,
    score_threshold: float = 0.8,
) -> NamedTuple("Outputs", [("deploy", str)]):
    
    
    """Evaluates an XGBoost model on a test dataset, logs metrics, and decides whether to deploy.

    Args:
        test_set: Input[Dataset]
            The test dataset as a Kubeflow component input.
        xgb_model: Input[Model]
            The trained XGBoost model as a Kubeflow component input.
        metrics: Output[ClassificationMetrics]
            A Kubeflow component output for logging classification metrics.
        smetrics: Output[Metrics]
            A Kubeflow component output for logging scalar metrics.
        bucket_name: str
            The name of the Google Cloud Storage bucket containing the model.
        score_threshold: float, default=0.8
            The minimum score required for deployment.

    Returns:
        NamedTuple("Outputs", [("deploy", str)])
            A named tuple with a single field:
            * deploy: str
                A string indicating whether to deploy the model ("true" or "false").
    """

    from google.cloud import storage
    import joblib
    import pandas as pd
    from sklearn.metrics import roc_curve, confusion_matrix
    from collections import namedtuple


    # ----- 1. Load Test Data and Model -----
    data = pd.read_csv(test_set.path)

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob_path = xgb_model.uri.replace(f"gs://{bucket_name}/", "")
    smetrics.log_metric("blob_path", str(blob_path))

    blob = bucket.blob(f"{blob_path}/model.joblib")
    with blob.open(mode="rb") as file:
        model = joblib.load(file)

    # ----- 2. Evaluation and Metrics -----
    y_scores = model.predict_proba(data.drop(columns=["will_buy_on_return_visit"]))[:, 1]
    y_pred = model.predict(data.drop(columns=["will_buy_on_return_visit"]))
    score = model.score(data.drop(columns=["will_buy_on_return_visit"]), data.will_buy_on_return_visit)

    fpr, tpr, thresholds = roc_curve(data.will_buy_on_return_visit.to_numpy(), y_scores, pos_label=True)
    metrics.log_roc_curve(fpr.tolist(), tpr.tolist(),thresholds.tolist())

    cm = confusion_matrix(data.will_buy_on_return_visit, y_pred)
    metrics.log_confusion_matrix(["False", "True"], cm.tolist())
    smetrics.log_metric("score", float(score))

    # ----- 3. Deployment Decision Logic -----
    deploy = "true" if score >= score_threshold else "false"

    # ----- 4. Metadata Update -----
    xgb_model.metadata["test_score"] = float(score)

    Outputs = namedtuple("Outputs", ["deploy"])
    return Outputs(deploy)


@dsl.component(base_image='python:3.8', 
    packages_to_install=["google-cloud-aiplatform==1.25.0"],
)
def deploy_xgboost_model(
    model: Input[Model],
    project_id: str,
    # region: str = "us-central1",  # Optional, default to a common region
    #endpoint_display_name: str = "breast-cancer-xgb-classification",
    #machine_type: str = "n1-standard-4",
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model],
) -> None:
    """Deploys an XGBoost model to Vertex AI Endpoint.

    Args:
        model: The model to deploy.
        project_id: The Google Cloud project ID.
        vertex_endpoint: Output[Artifact] representing the deployed Vertex AI Endpoint.
        vertex_model: Output[Model] representing the deployed Vertex AI Model.
    """

    from google.cloud import aiplatform

    # Initialize AI Platform with project 
    aiplatform.init(project=project_id)

    # Upload the Model
    deployed_model = aiplatform.Model.upload(
        display_name="xgb-classification",
        artifact_uri=model.uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest",
    )

    # Deploy the Model to an Endpoint
    endpoint = deployed_model.deploy(machine_type="n1-standard-4")

    # Save Outputs
    vertex_endpoint.uri = endpoint.resource_name
    vertex_model.uri = deployed_model.resource_name

    
@dsl.pipeline(
    # Default pipeline root. You can override it when submitting the pipeline.
    pipeline_root=PIPELINE_ROOT + "xgboost-pipeline-v2",
    # A name for the pipeline. Use to determine the pipeline Context.
    name="xgboost-pipeline-with-deployment-v2",
)
def pipeline():
    """
    Defines steps in pipeline
    """
    dataset_op = get_data(project_id = PROJECT_ID)
    training_op = train_model(dataset = dataset_op.outputs["dataset_train"])
    eval_op = eval_model(
        test_set=dataset_op.outputs["dataset_test"],
        xgb_model=training_op.outputs["model_artifact"],
        bucket_name = BUCKET_URI
    )

    with dsl.If(
        eval_op.outputs["deploy"] == "true",
        name="deploy",
    ):

        deploy_op = deploy_xgboost_model(model = training_op.outputs["model_artifact"],
                         project_id = PROJECT_ID,
                        )

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=pipeline, package_path="./xgb-pipeline.yaml"
    )

# import google.cloud.aiplatform as aiplatform
# import kfp
# from kfp import compiler, dsl
# from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output, component

# from kfp.dsl import ClassificationMetrics
# from typing import NamedTuple
# import os


# PROJECT_ID = os.environ['PROJECT_ID'] # replace with project ID
# REGION = os.environ['REGION']
# EXPERIMENT = 'vertex-pipelines'

# # gcs bucket
# GCS_BUCKET = PROJECT_ID
# BUCKET_URI = f"gs://{PROJECT_ID}-bucket"  # @param {type:"string"}
# PIPELINE_ROOT = PIPELINE_ROOT = BUCKET_URI + "/pipeline_root/"


# @component(
#     packages_to_install = [
#         "pandas==1.3.4",
#         "scikit-learn==1.0.1",
#     ],
# )
# def get_data(
#     dataset_train: Output[Dataset],
#     dataset_test: Output[Dataset],
# ):

#     from sklearn import datasets
#     from sklearn.model_selection import train_test_split as tts
#     import pandas as pd


#     # dataset https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
#     data_raw = datasets.load_breast_cancer()
#     data = pd.DataFrame(data_raw.data, columns=data_raw.feature_names)
#     data["target"] = data_raw.target

#     train, test = tts(data, test_size=0.3)

#     train.to_csv(dataset_train.path)
#     test.to_csv(dataset_test.path)


# @component(
#     packages_to_install = [
        
#         "xgboost==1.6.2",
#         "pandas==1.3.5",
#         "joblib==1.1.0",
#         "scikit-learn==1.0.2",
#         "pickle5==0.0.12",
#     ],
# )
# def train_model(
#     dataset: Input[Dataset],
#     model_artifact: Output[Model]
# ):

#     from xgboost import XGBClassifier
#     import pandas as pd
#     import pickle5 as pickle
#     import os
#     from pathlib import Path
#     import joblib
#     import logging
#     logging.basicConfig(level=logging.DEBUG)
    
    
#     data = pd.read_csv(dataset.path)

#     model = XGBClassifier(
#         objective="binary:logistic"
#     )
#     model.fit(
#         data.drop(columns=["target"]),
#         data.target,
#     )

#     score = model.score(
#         data.drop(columns=["target"]),
#         data.target,
#     )

#     model_artifact.metadata["train_score"] = float(score)
#     model_artifact.metadata["framework"] = "XGBoost"
#     os.makedirs(model_artifact.path, exist_ok=True)
#     joblib.dump(model, os.path.join(model_artifact.path, "model.joblib"))
#     print(model_artifact.path)
    

# @component(
#     packages_to_install = [
        
#         "xgboost==1.6.2",
#         "pandas==1.3.5",
#         "joblib==1.1.0",
#         "scikit-learn==1.0.2",
#         "google-cloud-storage==2.13.0",
#     ],
# )
# def eval_model(
#     test_set: Input[Dataset],
#     xgb_model: Input[Model],
#     metrics: Output[ClassificationMetrics],
#     smetrics: Output[Metrics],
#     bucket_name: str,
# ) -> NamedTuple("Outputs", [("deploy", str)]):
#     from xgboost import XGBClassifier
#     import pandas as pd

#     data = pd.read_csv(test_set.path)
#     model = XGBClassifier()
    
#     from google.cloud import storage
#     import joblib
#     client = storage.Client()
#     bucket = client.get_bucket(bucket_name)
#     blob_path = xgb_model.uri.replace("gs://" + bucket_name + "/", "")
#     smetrics.log_metric("blob_path", str(blob_path))

#     blob = bucket.blob(blob_path + "/model.joblib")
    
    
    
#     with blob.open(mode="rb") as file:
#         model = joblib.load(file)
#     #model.load_model(xgb_model.path)
    

#     score = model.score(
#         data.drop(columns=["target"]),
#         data.target,
#     )

#     from sklearn.metrics import roc_curve
#     y_scores =  model.predict_proba(data.drop(columns=["target"]))[:, 1]
#     fpr, tpr, thresholds = roc_curve(
#          y_true=data.target.to_numpy(), y_score=y_scores, pos_label=True
#     )
#     metrics.log_roc_curve(fpr.tolist(), tpr.tolist(), thresholds.tolist())

#     from sklearn.metrics import confusion_matrix
#     y_pred = model.predict(data.drop(columns=["target"]))

#     metrics.log_confusion_matrix(
#        ["False", "True"],
#        confusion_matrix(
#            data.target, y_pred
#        ).tolist()
#     )

#     xgb_model.metadata["test_score"] = float(score)
#     smetrics.log_metric("score", float(score))


#     deploy = "true"
#     #compare threshold or to previous

#     return (deploy,)



# @component(
#     packages_to_install=["google-cloud-aiplatform==1.25.0"],
# )
# def deploy_xgboost_model(
#     model: Input[Model],
#     project_id: str,
#     vertex_endpoint: Output[Artifact],
#     vertex_model: Output[Model],
# ):
#     """Deploys an XGBoost model to Vertex AI Endpoint.

#     Args:
#         model: The model to deploy.
#         project_id: The project ID of the Vertex AI Endpoint.

#     Returns:
#         vertex_endpoint: The deployed Vertex AI Endpoint.
#         vertex_model: The deployed Vertex AI Model.
#     """
#     from google.cloud import aiplatform

#     aiplatform.init(project=project_id)

#     deployed_model = aiplatform.Model.upload(
#         display_name="breastcancer-demo-model",
#         artifact_uri=model.uri,
#         serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest",
#     )
#     endpoint = deployed_model.deploy(machine_type="n1-standard-4")

#     vertex_endpoint.uri = endpoint.resource_name
#     vertex_model.uri = deployed_model.resource_name

    
    
# @dsl.pipeline(
#     # Default pipeline root. You can override it when submitting the pipeline.
#     pipeline_root=PIPELINE_ROOT + "xgboost-pipeline-v2",
#     # A name for the pipeline. Use to determine the pipeline Context.
#     name="xgboost-pipeline-with-deployment-v2",
# )
# def pipeline():
#     dataset_op = get_data()
#     training_op = train_model(dataset=dataset_op.outputs['dataset_train'
#                               ])
#     eval_op = eval_model(test_set=dataset_op.outputs['dataset_test'],
#                          xgb_model=training_op.outputs['model_artifact'
#                          ], bucket_name='kubeflow-mlops-410520-bucket')

#     with dsl.Condition(eval_op.outputs['deploy'] == 'true',
#                        name='deploy'):

#         deploy_op = \
#             deploy_xgboost_model(model=training_op.outputs['model_artifact'],
#                                  project_id=PROJECT_ID)

# if __name__ == '__main__':
#     compiler.Compiler().compile(
#         pipeline_func=pipeline, package_path="./breast-cancer-xgb-pipeline.yaml"
#     )