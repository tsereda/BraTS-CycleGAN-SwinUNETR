#!/bin/bash
echo "Creating temporary pod to fix permissions..."
kubectl run temp-fix-pod --image=busybox --restart=Never --overrides='
{
  "spec": {
    "containers": [
      {
        "name": "temp-fix-pod",
        "image": "busybox",
        "command": ["sleep", "3600"],
        "volumeMounts": [
          {
            "name": "brats-data-volume",
            "mountPath": "/data"
          }
        ]
      }
    ],
    "volumes": [
      {
        "name": "brats-data-volume",
        "persistentVolumeClaim": {
          "claimName": "brats-data"
        }
      }
    ]
  }
}'

echo "Waiting for pod to be ready..."
kubectl wait --for=condition=Ready pod/temp-fix-pod

echo "Creating output directory with proper permissions..."
kubectl exec temp-fix-pod -- mkdir -p /data/output
kubectl exec temp-fix-pod -- chmod 777 /data/output

echo "Verifying permissions..."
kubectl exec temp-fix-pod -- ls -la /data/output

echo "Cleaning up temporary pod..."
kubectl delete pod temp-fix-pod

echo "Done! You can now run your job."