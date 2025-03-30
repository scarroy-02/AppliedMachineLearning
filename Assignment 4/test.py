import os
import subprocess
import time
import requests
import pytest

def test_docker():
    # Check if container already exists and remove it
    try:
        subprocess.run(
            ["sudo", "docker", "rm", "-f", "flask-test-container"],
            check=False,  # Don't fail if container doesn't exist
            capture_output=True
        )
    except Exception:
        pass  # Ignore any errors during cleanup
    
    # Build the Docker image
    build_process = subprocess.run(
        ["sudo", "docker", "build", "-t", "flask-app", "."],
        check=True,
        capture_output=True,
        text=True
    )
    
    # Run the Docker container
    container_process = subprocess.Popen(
        ["sudo", "docker", "run", "-d", "-p", "5000:5000", "--name", "flask-test-container", "flask-app"],
        stdout=subprocess.PIPE,
        text=True
    )
    container_id = container_process.stdout.read().strip()
    
    try:
        # Wait for the container to start
        time.sleep(3)
        
        # Sample text to test
        sample_text = "This is a sample text for scoring"
        
        # Send request to the endpoint
        response = requests.post(
            "http://localhost:5000/score",
            json={"text": sample_text}
        )
        
        # Check if response is as expected
        assert response.status_code == 200
        # Add more specific assertions based on your app
        
    finally:
        # Stop and remove the container (use try/except to avoid test failures during cleanup)
        try:
            subprocess.run(["sudo", "docker", "stop", container_id], check=False)
            subprocess.run(["sudo", "docker", "rm", container_id], check=False)
        except Exception as e:
            print(f"Error during container cleanup: {e}")

if __name__ == "__main__":
    test_docker()