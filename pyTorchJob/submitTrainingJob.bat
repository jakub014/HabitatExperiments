set BUCKET_NAME=mytestbucketqpecs
set JOB_NAME=experiment_resnet_inception
set JOB_DIR=gs://%BUCKET_NAME%/%JOB_NAME%/models

gcloud ai-platform jobs submit training %JOB_NAME% \
  --region=us-central1 \
  --master-image-uri=gcr.io/cloud-ml-public/training/pytorch-gpu.1-4 \
  --scale-tier=CUSTOM \
  --master-machine-type=n1-standard-8 \
  --master-accelerator=type=nvidia-tesla-p100,count=1 \
  --job-dir=%JOB_DIR% \
  --package-path=./trainer \
  --module-name=trainer.task