# S3 Vectors

The S3 Vectors storage provider uses AWS S3 Vector Buckets to store and query vector indexes.

## Requirements

To use the S3 Vectors storage provider you must:

-   Have an AWS account with an S3 Vector Bucket configured
-   Have the `boto3` Python package installed
-   AWS access credentials in your application environment

## Usage

Instantiate a S3 Vectors provider with:

```python
from django_ai_core.contrib.index.storage.s3vectors import S3VectorProvider

storage_provider = S3VectorProvider(
    bucket_name="my-bucket-name", dimensions=1536
)
```

Where `bucket_name` is the name of your pre-existing S3 Vector Bucket and `dimensions` is the number of dimensions output by your selected embedding model.

If using this storage provider outside of the context of a `VectorIndex`, you will also need to specify an `index_name`.
