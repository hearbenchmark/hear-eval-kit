Instructions for using Spotty
=============================
### Setup GCP
#### Create Project
To use spotty, please sign up on GCP and 
[create a project](https://console.cloud.google.com/projectcreate) if you dont have one.
Please refer [here](https://spotty.cloud/docs/providers/gcp/account-preparation.html) for more details on GCP Account Preperation for spotty.

#### Service account Key
[Create a service account](https://console.cloud.google.com/iam-admin/serviceaccounts/create) and 
generate a key for the account. Download the key and add the path
to the environment variable:
```
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/the/service/account/key/file.json"
```

### Configure gs-util:
`gs-util` helps to sync the local files to the instance. Ensure gsutil is installed (already present in requirements) and configure it:
```
gsutil config -f
```
Running the above command will prompt to follow a link to authorise 
with the gmail account used for google cloud. Please copy the 
`authorisation code` from this browser back to the console. It will 
also ask for the `project id string`, which can be found in the project seletion dropdown on google cloud console.

Please refer [here](https://cloud.google.com/storage/docs/gsutil_install#install) 
for more details on gsutil installation.

### Prepare spotty.yaml file
Copy over the `spotty.yaml.tmpl` file to generate a `spotty.yaml` file
```
cp spotty.yaml.tmpl spotty.yaml
```

Change the instance name in the copied file. Specifically, change `"USERNAME"` suffix 
in `instances: name` to allow for multiple users in the same project to 
make separate gcp instances and volumes to avoid conflicts within the project.

### Run spotty
Run the below command:
```
spotty start
spotty sh
```
It runs a tmux session, so you can always detach this session using 
`Ctrl + b`, then `d` combination of keys. To be attached to that session 
later, just use the spotty sh command again.

The `spotty.yaml` is configured to automatically
- Create a gcp cloud instance by the name `spotty-heareval-i1`
- Associate disks to the instance by the name `spotty-heareval-spotty-heareval-i1-workspace`
- Setup CUDA (Without any hassle)
- Sync the local files to the instance
- Install the docker image for heareval

### Terminate spotty
After the evaluation is complete, the instance can be terminated by:
```
spotty stop
```
Delete the disk(volume) associated with the instance after the evaluation is complete. 
Automatic Disk Deletion policy is not supported by spotty for GCP, 
this has to be done manually [here](https://console.cloud.google.com/compute/disks)

Disks can also be deleted from cli. Please ensure 
[gcloud](https://cloud.google.com/sdk/docs/install) is installed and configured 
before running the below command for deleting disks:
```
gcloud compute disks delete disk-name --zone=zone
```
The default name of the disk created by `spotty start` here is `spotty-heareval-spotty-
heareval-i1-workspace`