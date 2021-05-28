"""
Utility functions for interacting with an S3 bucket.
"""

import os
import subprocess

import boto3
from luigi.contrib.s3 import S3Client
from botocore.client import ClientError
import luigi

from .luigi import WorkTask


class CacheTarCorpus(WorkTask):
    """
    If the tar file is cached in S3, we simply retrieve it and untar
    it, and skip the pipeline.

    If the tar file is NOT in S3, we run the pipeline, create the
    tar-file, and upload it to S3.
    """

    task_name = luigi.Parameter()
    bucket = luigi.Parameter()
    region = luigi.Parameter()
    next_task = luigi.Parameter()
    # TODO: How do we support params for next_task??

    def requires(self):
        return EnsureBucket(self.bucket, self.region)

    @property
    def name(self):
        return type(self).__name__

    def run(self):
        tarfile = f"{self.task_name}.tar.gz"
        pathtarfile = f"{self.workdir}/{tarfile}"
        client = S3Client()
        s3cache = os.path.join(f"s3://{self.bucket}/", tarfile)
        if client.exists(s3cache):
            # TODO: Create a new task to untar this a move it out of
            # workdir into the main evaluation dir so it is good to go.
            client.get(s3cache, pathtarfile)
        else:
            # If you yield FinalizeCorpus, this task is suspended
            # and FinalizeCorpus is run, and a LocalFileTarget is
            # returned.
            finalize_corpus = yield self.next_task()

            # Tar the file
            devnull = open(os.devnull, "w")
            ret = subprocess.call(
                [
                    "tar",
                    "zcvf",
                    pathtarfile,
                    # Unfortunately, we have to hardcode
                    # FinalizeCorpus.workdir()
                    # because we don't have a Task
                    # just a Target in finalize_corpus.
                    self.task_name,
                ],
                stdout=devnull,
                stderr=devnull,
            )
            # Make sure the return code is 0 and the command was successful.
            assert ret == 0

            # Cache to S3
            print("Putting file to S3")
            client.put_multipart(pathtarfile, s3cache)

        with self.output().open("w") as outfile:
            pass


class EnsureBucket(WorkTask):
    """
    Ensure the S3 bucket exists and is readable.

    This S3 code is pretty gnarly, but I'm not sure it can be made
    any cleaner.
    """

    bucket = luigi.Parameter()
    region = luigi.Parameter()

    @property
    def name(self):
        return type(self).__name__

    def run(self):
        check_bucket(self.bucket, self.region)
        with self.output().open("w") as outfile:
            pass


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

    # TODO: test this when region is set to None and update README on how that works.
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
