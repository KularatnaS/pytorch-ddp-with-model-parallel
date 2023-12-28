from sagemaker.pytorch import PyTorch
import sagemaker

sagemaker_session = sagemaker.Session()
role = 'arn:aws:iam::101245251104:role/service-role/AmazonSageMaker-ExecutionRole-20210115T112099'

pt_estimator = PyTorch(
    entry_point="run.py",
    source_dir="source",
    role=role,
    framework_version="2.1.0",
    py_version="py310",
    instance_count=1,
    instance_type="ml.g4dn.12xlarge",
    distribution={
        "torch_distributed": {
            "enabled": True
        }
    }
)

pt_estimator.fit()