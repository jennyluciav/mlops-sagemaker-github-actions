import sys
import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-Uq', 'pip'])
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sagemaker==2.73.0', '-Uq'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sagemaker-experiments==0.1.35', '-Uq'])

import boto3
import os
import pandas as pd
import sagemaker
import time
from datetime import datetime
from sagemaker import ModelPackage
from sagemaker.image_uris import retrieve
from sagemaker.lambda_helper import Lambda
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.callback_step import CallbackStep
from sagemaker.workflow.lambda_step import LambdaStep
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, CacheConfig
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
# Configure an Experiment
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments.trial_component import TrialComponent
from sagemaker.workflow.pipeline import PipelineExperimentConfig


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags

def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="AutoPilotPackageGroup",
    pipeline_name="AutoPilotPipeline",
    base_job_prefix="Autopilot",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
):
    ## Initialization
    boto_session = boto3.session.Session()
    aws_region = boto_session.region_name
    sagemaker_client = boto_session.client("sagemaker")
    lambda_client = boto_session.client("lambda")
    sagemaker_session = sagemaker.session.Session(
        boto_session=boto_session, sagemaker_client=sagemaker_client
    )
    sqs_client = boto_session.client(
        "sqs",
        region_name=aws_region,
        endpoint_url=f"https://sqs.{aws_region}.amazonaws.com",
    )

    experiment_name = 'Autopilot-sagemaker-experiments-pipelines'

    ## IAM PERMISSIONS
    lambda_execution_role_name = "LAMBDA_EXECUTION_ROLE_SAGEMAKER_AUTOPILOT"
    aws_account_id = boto3.client("sts").get_caller_identity().get("Account")
    LAMBDA_EXECUTION_ROLE_ARN = f"arn:aws:iam::{aws_account_id}:role/{lambda_execution_role_name}"  # to be assumed by the Lambda service
    SAGEMAKER_EXECUTION_ROLE_ARN = (
        sagemaker.get_execution_role()
    )  # to be assumed by the SageMaker service
    PROCESSING_JOB_LOCAL_BASE_PATH = "/opt/ml/processing"

    ## SageMaker Pipelines parameters
    cache_config = CacheConfig(enable_caching=False)
    autopilot_job_name = ParameterString(
        name="AutopilotJobName",
        default_value="autopilot-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    )
    model_package_name = ParameterString(
        name="ModelPackageName",
        default_value=autopilot_job_name.default_value + "-model-package",
    )
    target_attribute_name = ParameterString(name="TargetAttributeName", default_value="readmitted")
    train_val_dataset_s3_path = ParameterString(
        name="TrainValDatasetS3Path",
        default_value=sagemaker.s3.s3_path_join(
            "s3://", default_bucket, "odilo", "data", "train_val.csv"
        ),
    )
    x_test_s3_path = ParameterString(
        name="XTestDatasetS3Path",
        default_value=sagemaker.s3.s3_path_join(
            "s3://", default_bucket, "odilo", "data", "x_test.csv"
        ),
    )
    y_test_s3_path = ParameterString(
        name="YTestDatasetS3Path",
        default_value=sagemaker.s3.s3_path_join(
            "s3://", default_bucket, "odilo", "data", "y_test.csv"
        ),
    )
    max_autopilot_candidates = ParameterInteger(name="MaxAutopilotCandidates", default_value=16)
    max_autopilot_job_runtime = ParameterInteger(
        name="MaxAutoMLJobRuntimeInSeconds", default_value=7200  # 2 hours
    )
    max_autopilot_training_job_runtime = ParameterInteger(
        name="MaxRuntimePerTrainingJobInSeconds", default_value=3600
    )  # 1 hour
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="Approved")
    batch_transform_output_s3_path = ParameterString(
        name="BatchTransformOutputS3Path",
        default_value=sagemaker.s3.s3_path_join(
            "s3://", default_bucket, autopilot_job_name.default_value, "batch-transform-output"
        ),
    )
    training_output_s3_path = ParameterString(
        name="TrainingOutputS3Path",
        default_value=sagemaker.s3.s3_path_join(
            "s3://", default_bucket, autopilot_job_name.default_value, "training-output"
        ),
    )

    ## First pipeline step: start Autopilot job
    lambda_start_autopilot_job = Lambda(
        function_name="StartSagemakerAutopilotJob",
        execution_role_arn=LAMBDA_EXECUTION_ROLE_ARN,
        script="start_autopilot_job.py",
        handler="start_autopilot_job.lambda_handler",
        session=sagemaker_session,
    )
    lambda_start_autopilot_job.upsert()
    step_start_autopilot_job = LambdaStep(
        name="StartAutopilotJobStep",
        lambda_func=lambda_start_autopilot_job,
        inputs={
            "TrainValDatasetS3Path": train_val_dataset_s3_path.default_value,
            "MaxCandidates": max_autopilot_candidates.default_value,
            "MaxRuntimePerTrainingJobInSeconds": max_autopilot_training_job_runtime.default_value,
            "MaxAutoMLJobRuntimeInSeconds": max_autopilot_job_runtime.default_value,
            "TargetAttributeName": target_attribute_name.default_value,
            "TrainingOutputS3Path": training_output_s3_path.default_value,
            "AutopilotJobName": autopilot_job_name,
            "ProblemType": "MulticlassClassification",
            "AutopilotExecutionRoleArn": SAGEMAKER_EXECUTION_ROLE_ARN,
            "AutopilotObjectiveMetricName": "F1macro",
            "AutopilotMode": "ENSEMBLING",
        },
        cache_config=cache_config,
    )

    ## Second pipeline step: check Autopilot job status
    lambda_check_autopilot_job_status_function_name = "CheckSagemakerAutopilotJobStatus"
    lambda_check_autopilot_job_status = Lambda(
        function_name=lambda_check_autopilot_job_status_function_name,
        execution_role_arn=LAMBDA_EXECUTION_ROLE_ARN,
        script="check_autopilot_job_status.py",
        handler="check_autopilot_job_status.lambda_handler",
        session=sagemaker_session,
        timeout=15,
    )
    lambda_check_autopilot_job_status.upsert()
    queue_url = sqs_client.create_queue(
        QueueName="AutopilotSagemakerPipelinesSqsCallback",
        Attributes={"DelaySeconds": "5", "VisibilityTimeout": "300"},
    )[
        "QueueUrl"
    ]  # 5 minutes timeout
    # Add event source mapping
    try:
        response = lambda_client.create_event_source_mapping(
            EventSourceArn=sqs_client.get_queue_attributes(
                QueueUrl=queue_url, AttributeNames=["QueueArn"]
            )["Attributes"]["QueueArn"],
            FunctionName=lambda_check_autopilot_job_status_function_name,
            Enabled=True,
            BatchSize=1,
        )
    except lambda_client.exceptions.ResourceConflictException:
        pass
    step_check_autopilot_job_status_callback = CallbackStep(
        name="CheckAutopilotJobStatusCallbackStep",
        sqs_queue_url=queue_url,
        inputs={"AutopilotJobName": autopilot_job_name},
        outputs=[],
        depends_on=[step_start_autopilot_job],
    )

    ## Third pipeline step: evaluate Autopilot model
    processing_evaluation = SKLearnProcessor(
        role=SAGEMAKER_EXECUTION_ROLE_ARN,
        framework_version="1.0-1",
        instance_count=instance_count.default_value,
        instance_type=instance_type.default_value,
        sagemaker_session=sagemaker_session,
    )
    step_autopilot_model_evaluation = ProcessingStep(
        name="EvaluateBestAutopilotModelStep",
        job_arguments=[
            "--autopilot-job-name",
            autopilot_job_name,
            "--aws-region",
            aws_region,
            "--batch-transform-output-s3-path",
            batch_transform_output_s3_path.default_value,
            "--instance-type",
            instance_type.default_value,
            "--instance-count",
            str(instance_count.default_value),
            "--local-base-path",
            PROCESSING_JOB_LOCAL_BASE_PATH,
            "--sagemaker-execution-role-arn",
            SAGEMAKER_EXECUTION_ROLE_ARN,
            "--x-test-s3-path",
            x_test_s3_path.default_value,
            "--y-test-file-name",
            y_test_s3_path.default_value.split("/")[-1],
        ],
        processor=processing_evaluation,
        code="evaluate_autopilot_model.py",
        depends_on=[step_check_autopilot_job_status_callback],
        inputs=[
            ProcessingInput(
                input_name="LabelsTestDataset",
                source=y_test_s3_path.default_value,
                destination=os.path.join(PROCESSING_JOB_LOCAL_BASE_PATH, "data"),
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="EvaluationReport",
                source=os.path.join(PROCESSING_JOB_LOCAL_BASE_PATH, "evaluation_report"),
            )
        ],
        cache_config=cache_config,
    )

    ## Fourth pipeline step: register Autopilot model
    lambda_register_autopilot_model = Lambda(
        function_name="RegisterSagemakerAutopilotModel",
        execution_role_arn=LAMBDA_EXECUTION_ROLE_ARN,
        script="register_autopilot_model.py",
        handler="register_autopilot_model.lambda_handler",
        session=sagemaker_session,
        timeout=15,
    )
    lambda_register_autopilot_model.upsert()
    step_register_autopilot_model = LambdaStep(
        name="RegisterAutopilotModelStep",
        lambda_func=lambda_register_autopilot_model,
        inputs={
            "AutopilotJobName": autopilot_job_name,
            "EvaluationReportS3Path": step_autopilot_model_evaluation.properties.ProcessingOutputConfig.Outputs[
                "EvaluationReport"
            ].S3Output.S3Uri,
            "ModelPackageName": model_package_name.default_value,
            "ModelApprovalStatus": model_approval_status.default_value,
            "InstanceType": instance_type.default_value,
        },
        cache_config=cache_config,
    )

    # create an experiment if it doesnt exist
    try:
        demo_experiment = Experiment.load(experiment_name=experiment_name)
        print("existing experiment loaded")
    except Exception as ex:
        if "ResourceNotFound" in str(ex):
            demo_experiment = Experiment.create(
                experiment_name=experiment_name,
                description = "Demo experiment",
                tags = [{'Key': 'demo-experiments', 'Value': 'demo1'}]
            )
            print("new experiment created")
        else:
            print(f"Unexpected {ex}, {type(ex)}")
            print("Dont go forward!")
            raise

    pipeline_name = f"AutoPilotExperimentsPipeline"

    #Pipeline experiment config
    autopilot_experiment_config = PipelineExperimentConfig(
        experiment_name,
        Join(
            on="-",
            values=[
                "pipeline-execution",
                ExecutionVariables.PIPELINE_EXECUTION_ID
            ],
        )
    )

    ## Create and run pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        pipeline_experiment_config=autopilot_experiment_config,
        parameters=[
            autopilot_job_name,
            target_attribute_name,
            train_val_dataset_s3_path,
            x_test_s3_path,
            y_test_s3_path,
            max_autopilot_candidates,
            max_autopilot_job_runtime,
            max_autopilot_training_job_runtime,
            instance_count,
            instance_type,
            model_approval_status,
        ],
        steps=[
            step_start_autopilot_job,
            step_check_autopilot_job_status_callback,
            step_autopilot_model_evaluation,
            step_register_autopilot_model,
        ],
        sagemaker_session=sagemaker_session,
    )
    #pipeline.upsert(role_arn=SAGEMAKER_EXECUTION_ROLE_ARN)
    #pipeline_execution = pipeline.start()
    return pipeline
    