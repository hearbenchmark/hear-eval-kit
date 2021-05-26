"""
Configuration specific to AWS S3
"""

# If set to true will cache results in S3
S3_CACHE = True

# You should pick a unique handle, since this determine the S3 path
# (which must be globally unique across all S3 users).
HANDLE = "jshier"
S3_BUCKET = f"hear2021-{HANDLE}"

# If this is None, boto will use whatever is in your
# ~/.aws/config or AWS_DEFAULT_REGION environment variable
S3_REGION_NAME = "us-west-2"
