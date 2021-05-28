"""
Configuration specific to AWS S3
"""

# You should pick a unique handle, since this determine the S3 path
# (which must be globally unique across all S3 users).
HANDLE = "hear"
S3_BUCKET = f"hear2021-{HANDLE}"

# If this is None, boto will use whatever is in your
# ~/.aws/config or AWS_DEFAULT_REGION environment variable
S3_REGION_NAME = "eu-central-1"
