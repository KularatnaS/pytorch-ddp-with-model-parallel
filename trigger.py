from sagemaker.pytorch import PyTorch
import sagemaker

mpi_options = "-verbose --mca orte_base_help_aggregate 0 "
smp_parameters = {
    "ddp": True,
    "fp16": True,
    "prescaled_batch": True,
    "sharded_data_parallel_degree": 1,
    "tensor_parallel_degree": 2
}

pytorch_estimator = PyTorch(
    source_dir="source",
    entry_point="run.py",
    role=sagemaker.get_execution_role(),
    instance_type="ml.p4d.24xlarge",
    volume_size=200,
    instance_count=2,
    py_version="py39",
    framework_version="1.13.1",
    distribution={
        "smdistributed": {
            "modelparallel": {
                "enabled": True,
                "parameters": smp_parameters,
            }
        },
        "mpi": {
            "enabled": True,
            "processes_per_host": 8,
            "custom_mpi_options": mpi_options,
        },
    },
)


pytorch_estimator.fit()