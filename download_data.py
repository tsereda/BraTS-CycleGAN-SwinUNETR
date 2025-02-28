import kagglehub

# Download latest version
path = kagglehub.dataset_download("awsaf49/brats2020-training-data", "./raw_data/brats")

print("Path to dataset files:", path)