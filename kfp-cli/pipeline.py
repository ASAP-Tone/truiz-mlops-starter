import google.cloud.aiplatform as aiplatform
import kfp
from kfp import compiler, dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output, component

from kfp.dsl import ClassificationMetrics
from typing import NamedTuple
import os

PROJECT_ID = 'kubeflow-mlops-410520' # replace with project ID
REGION = 'us-central1'
EXPERIMENT = 'vertex-pipelines'
SERIES = 'dev'

# gcs bucket
GCS_BUCKET = PROJECT_ID
BUCKET_URI = f"gs://{PROJECT_ID}-bucket"  # @param {type:"string"}
PIPELINE_ROOT = PIPELINE_ROOT = BUCKET_URI + "/pipeline_root/"


@component(
    packages_to_install = [
        "pandas==1.3.4",
        "scikit-learn==1.0.1",
    ],
)
def get_data(
    dataset_train: Output[Dataset],
    dataset_test: Output[Dataset],
):

    from sklearn import datasets
    from sklearn.model_selection import train_test_split as tts
    import pandas as pd


    # dataset https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
    data_raw = datasets.load_breast_cancer()
    data = pd.DataFrame(data_raw.data, columns=data_raw.feature_names)
    data["target"] = data_raw.target

    train, test = tts(data, test_size=0.3)

    train.to_csv(dataset_train.path)
    test.to_csv(dataset_test.path)


@component(
    packages_to_install = [
        
        "xgboost==1.6.2",
        "pandas==1.3.5",
        "joblib==1.1.0",
        "scikit-learn==1.0.2",
        "pickle5==0.0.12",
        "joblib==1.1.0",
    ],
)
def train_model(
    dataset: Input[Dataset],
    model_artifact: Output[Model]
):

    from xgboost import XGBClassifier
    import pandas as pd
    import pickle5 as pickle
    import os
    from pathlib import Path
    import joblib
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    
    data = pd.read_csv(dataset.path)

    model = XGBClassifier(
        objective="binary:logistic"
    )
    model.fit(
        data.drop(columns=["target"]),
        data.target,
    )

    score = model.score(
        data.drop(columns=["target"]),
        data.target,
    )

    model_artifact.metadata["train_score"] = float(score)
    model_artifact.metadata["framework"] = "XGBoost"
    os.makedirs(model_artifact.path, exist_ok=True)
    joblib.dump(model, os.path.join(model_artifact.path, "model.joblib"))
    print(model_artifact.path)
    

@component(
    packages_to_install = [
        
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
) -> NamedTuple("Outputs", [("deploy", str)]):
    from xgboost import XGBClassifier
    import pandas as pd

    data = pd.read_csv(test_set.path)
    model = XGBClassifier()
    
    from google.cloud import storage
    import joblib
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob_path = xgb_model.uri.replace("gs://" + bucket_name + "/", "")
    smetrics.log_metric("blob_path", str(blob_path))

    blob = bucket.blob(blob_path + "/model.joblib")
    
    
    
    with blob.open(mode="rb") as file:
        model = joblib.load(file)
    #model.load_model(xgb_model.path)
    

    score = model.score(
        data.drop(columns=["target"]),
        data.target,
    )

    from sklearn.metrics import roc_curve
    y_scores =  model.predict_proba(data.drop(columns=["target"]))[:, 1]
    fpr, tpr, thresholds = roc_curve(
         y_true=data.target.to_numpy(), y_score=y_scores, pos_label=True
    )
    metrics.log_roc_curve(fpr.tolist(), tpr.tolist(), thresholds.tolist())

    from sklearn.metrics import confusion_matrix
    y_pred = model.predict(data.drop(columns=["target"]))

    metrics.log_confusion_matrix(
       ["False", "True"],
       confusion_matrix(
           data.target, y_pred
       ).tolist()
    )

    xgb_model.metadata["test_score"] = float(score)
    smetrics.log_metric("score", float(score))


    deploy = "true"
    #compare threshold or to previous

    return (deploy,)



@component(
    packages_to_install=["google-cloud-aiplatform==1.25.0"],
)
def deploy_xgboost_model(
    model: Input[Model],
    project_id: str,
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model],
):
    """Deploys an XGBoost model to Vertex AI Endpoint.

    Args:
        model: The model to deploy.
        project_id: The project ID of the Vertex AI Endpoint.

    Returns:
        vertex_endpoint: The deployed Vertex AI Endpoint.
        vertex_model: The deployed Vertex AI Model.
    """
    from google.cloud import aiplatform

    aiplatform.init(project=project_id)

    deployed_model = aiplatform.Model.upload(
        display_name="census-demo-model",
        artifact_uri=model.uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest",
    )
    endpoint = deployed_model.deploy(machine_type="n1-standard-4")

    vertex_endpoint.uri = endpoint.resource_name
    vertex_model.uri = deployed_model.resource_name

    
    
@dsl.pipeline(
    # Default pipeline root. You can override it when submitting the pipeline.
    pipeline_root=PIPELINE_ROOT + "xgboost-pipeline-v2",
    # A name for the pipeline. Use to determine the pipeline Context.
    name="xgboost-pipeline-with-deployment-v2",
)
def pipeline():
    dataset_op = get_data()
    training_op = train_model(dataset=dataset_op.outputs['dataset_train'
                              ])
    eval_op = eval_model(test_set=dataset_op.outputs['dataset_test'],
                         xgb_model=training_op.outputs['model_artifact'
                         ], bucket_name='kubeflow-mlops-410520-bucket')

    with dsl.Condition(eval_op.outputs['deploy'] == 'true',
                       name='deploy'):

        deploy_op = \
            deploy_xgboost_model(model=training_op.outputs['model_artifact'
                                 ], project_id=PROJECT_ID)


if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=pipeline, package_path="breast-cancer-xgb-pipeline.yaml"
    )