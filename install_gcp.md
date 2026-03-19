# Google Cloud Platform (GCP) GPU Instance Setup Guide

This guide will help you set up a Google Cloud GPU instance with the latest OS and Python versions, including manual installation of NVIDIA drivers and CUDA Toolkit.

---

## 1. Install the Google Cloud CLI

Follow the official instructions: [Install gcloud CLI](https://cloud.google.com/sdk/docs/install)

Verify installation:
```sh
gcloud --version
```
Example output:
```
Google Cloud SDK 527.0.0
bq 2.1.18
core 2025.06.13
gcloud-crc32c 1.0.0
gsutil 5.34
```

---

## 2. Authenticate and Set Up Your Project

Authenticate with your Google account:
```sh
gcloud auth login
```

List your available projects (create one in the console if needed):
```sh
gcloud projects list
```

Set your active project (replace with your project ID):
```sh
gcloud config set project sp24-422822
```

---

## 3. GPU Quota and Instance Creation

- **Increase your GPU quota**: By default, `GPUS_ALL_REGIONS` is 0. Request a quota increase here: [GCP Quotas](https://cloud.google.com/compute/quotas)
- **Create a VM instance** (try a different zone if resources are unavailable):

```sh
gcloud compute instances create gpu-instance \
    --zone=us-west2-c \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --boot-disk-size=200GB \
    --maintenance-policy=TERMINATE
```

Expected output:
```
NAME          ZONE        MACHINE_TYPE   PREEMPTIBLE  INTERNAL_IP  EXTERNAL_IP    STATUS
gpu-instance  us-west2-c  n1-standard-4               10.168.0.3   35.235.85.236  RUNNING
```

---

## 4. Connect to Your Instance

SSH into your instance:
```sh
gcloud compute ssh gpu-instance --zone=us-west2-c
```

---

## 5. Install GPU Drivers and CUDA Toolkit

Follow the official guide: [Install GPU drivers on Linux](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#linux)

- The process may take a few minutes and may prompt you to restart the instance.
- **Verify** both the driver and CUDA installation as described in the guide.

> **Note:** As of June 2025, the versions are:  
> **Driver Version:** 575.57.08  
> **CUDA Version:** 12.9

---

## 6. Shutdown When Not in Use

To avoid unnecessary charges, always shut down your GPU instance when you are not actively using it. Google Cloud bills for running instances, even if you are not connected.

You have several options to safely stop your instance:

- **Via GCP Console:**
  - Go to the [Google Cloud Console](https://console.cloud.google.com/compute/instances), select your instance, and click **Stop**.

- **From your local machine:**
  - Use the following command (replace the zone and instance name if different):
    ```sh
    gcloud compute instances stop gpu-instance --zone=us-west2-c
    ```

- **From within the instance (SSH session):**
  - Run:
    ```sh
    sudo shutdown -h now
    ```

> **Tip:** Stopping an instance preserves your data on the boot disk, but you are still charged for the disk and any reserved IP addresses. To avoid all charges, consider deleting the instance and associated resources when finished.

---

## Additional Tips
- Always monitor your usage and billing in the GCP console.
- For troubleshooting, refer to the [GCP documentation](https://cloud.google.com/docs/).