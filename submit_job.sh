#!/bin/bash

# Helper script to submit GPT-2 inference job and monitor it

echo "=== GPT-2 Inference Job Submission ==="
echo "Submitting job..."

# Submit the job and capture the job ID
JOB_OUTPUT=$(qsub inference.pbs)
if [ $? -eq 0 ]; then
    JOB_ID=$(echo $JOB_OUTPUT | grep -o '[0-9]*')
    echo "Job submitted successfully!"
    echo "Job ID: $JOB_ID"
    echo ""
    
    # Create logs directory if it doesn't exist
    mkdir -p logs
    
    echo "=== Monitoring Job Status ==="
    echo "You can monitor the job using:"
    echo "  qstat -f $JOB_ID"
    echo "  qstat -u \$USER"
    echo ""
    
    echo "=== Log Files ==="
    echo "Live output log: logs/${JOB_ID}.log"
    echo "Error log: logs/${JOB_ID}_error.log"
    echo "PBS output: output_${JOB_ID}.log"
    echo ""
    
    echo "=== Tail Live Logs ==="
    echo "To follow the live logs, use:"
    echo "  tail -f logs/${JOB_ID}.log"
    echo "  tail -f logs/${JOB_ID}_error.log"
    echo ""
    
    # Wait a moment and show initial status
    sleep 2
    echo "Current job status:"
    qstat $JOB_ID
    
else
    echo "ERROR: Failed to submit job!"
    echo "Please check your PBS script and try again."
    exit 1
fi
