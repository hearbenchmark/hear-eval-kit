"""
Utility functions for interacting with an S3 bucket.
"""

import boto3
from luigi.contrib.s3 import S3Client
from botocore.client import ClientError

from .luigi import WorkTask


def can_access_bucket(s3: boto3.session.Session, bucket: str) -> str:
    """
    Checks access to bucket

    Args:
        s3: s3 connection
        bucket: name of bucket to check

    Returns:
        boolean indicating whether bucket can be accessed

    """
    try:
        s3.head_bucket(Bucket=bucket)
        return True
    except ClientError:
        return False


def check_bucket(bucket: str, region: str):
    """
    Attempt to access an S3 bucket. If it doesn't exist,
    then it is created. An error is raised if the operation
    was unsuccessful.

    Args:
        bucket: name of bucket
        region: region the bucket exists in
    """

    if region is None:
        s3 = boto3.client("s3")
    else:
        s3 = boto3.client("s3", region_name=region)

    # If cannot access bucket, try to create it.
    if not can_access_bucket(s3, bucket):
        # Try to create the bucket
        if region is None:
            bucket = s3.create_bucket(Bucket=bucket)
        else:
            location = {"LocationConstraint": region}
            bucket = s3.create_bucket(Bucket=bucket, CreateBucketConfiguration=location)

    # Make sure we can access it
    try:
        can_access_bucket(s3, bucket)
    except ClientError:
        raise ValueError(
            f"S3 bucket {bucket} does not exist or you don't have access. "
            "Have you changed HANDLE to something unique to you?"
        )
