# Image Classification using AWS SageMaker

AWS Sagemaker project to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and allowing queries to the best model via a deployed endpoint. This is done on the dog breed classication data set.

![Dog Predictions](Screenshots/dogPredictions.png)


## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the dataset from s3 bucket. 

## Dataset
The dataset is the dogbreed classification dataset which has 133 classes of images of dogs.

### Access
Dataset can be downloaded using the script from the s3 bucket.

## Hyperparameter Tuning
The model chosen was the pretrained Resnet50, which was frozen for training. A fully connected neural network on is then placed behind this to facilitate image classification. The hyperparameters of the fully connected neural network are then tune to optimize performance and minimize loss. Hyperparameters chosen to tune are: 
- lr as in range 0.001 to 0.1. Tested as this is a reasonable range to test without having too severe a learning rate. Optimal value: 0.0014611377532535268
- batch-size between 32, 64, 128. Range of values as balance between accuracy and speed. Optimal value: 64 
- epochs between 2 and 4. Balance again between accuracy and speed. Optimal value: 3

![Hyperparameter Training](./Screenshots/HPTuningJobs2.png)

## Debugging and Profiling
Debugging and profiling was used with the following rules are included here: [Profiler Report](./ProfilerReport/profiler-output/profiler-report.html)
. Included here is a brief summary:
### Step Duration Analysis
- StepOutlier Rule: Flags steps with durations over 3 times the standard deviation.
-- Parameters: Mode set to None; ignores the first 10 outliers.
-- Data Analyzed: 20,042 datapoints, with 5 triggers.
- Node algo-1-33 Summary:
-- Mean Duration: 0.0s, with 2 outliers over 3 times the standard deviation of 0.1s.
-- Duration Stats: Mean: 0.00s, Max: 13.96s, p99: 0.04s, p95: 0.00s, p50: 0.00s, Min: 0.00s.
### GPU Utilization Analysis
- LowGPUUtilization Rule: Checks for low or fluctuating GPU usage.
-- Analysis: 95th percentile >70%, 5th percentile <10% in 2 cases.
-- Recommendation: Consider a smaller instance or increased batch size.
-- Recent Trigger: 11/13/2024 at 13:24:00.
- Node algo-1 GPU0 Utilization:
--Utilization Stats: Max 99.0%, 5th percentile 0.0%, indicating significant fluctuation.
### Workload Balancing
- LoadBalancing Rule: Checks for workload distribution across GPUs.
-- Result: Only one GPU used; no workload balancing issues detected.
### Dataloading Analysis
- Dataloader Workers: Assesses the number of dataloading processes.
-- Result: 0 datapoints analyzed; no issues triggered.
### Batch Size
- BatchSize Rule: Detects underutilization due to small batch size.
-- Analysis: Checked GPU memory, CPU, and GPU utilization over a 500-datapoint window.
-- Result: No underutilization detected.
### CPU Bottlenecks
- CPUBottleneck Rule: Monitors CPU >90% and GPU <10% usage.
-- Result: 305 bottlenecks found (24% of total time), below the 50% threshold.
### I/O Bottlenecks
- IOBottleneck Rule: Monitors I/O wait time >50% and GPU <10%.
-- Result: 44 bottlenecks (3% of total time), below the 50% threshold.
### GPU Memory
- GPUMemoryIncrease Rule: Detects large increases in GPU memory usage.
-- Analysis: Triggered 4 times with memory spikes.
-- Recent Trigger: 11/13/2024 at 13:24:00.
- Node algo-1 GPU0 Memory:
-- Memory Utilization Stats: Max 75.0%, 5th percentile 0.0%, with significant fluctuation (95th percentile at 58%).

## Model Deployment
Model with best hyperparameters is deployed as an endpoint with a single initial 'ml.m5.large'. It is possible to query the endpoint with an image, providing that it is converted to a byte array first. 

![Endpoint operation](./Screenshots/Endpoint_operation.png)

