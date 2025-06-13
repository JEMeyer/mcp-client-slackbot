import time
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Set environment variables or edit the corresponding values here.
endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
api_key = os.getenv('AZURE_OPENAI_API_KEY')

api_version = 'preview'
headers= { "api-key": api_key, "Content-Type": "application/json" }

def create_video_job(prompt, width=480, height=480, n_seconds=5, model="sora"):
    """Create a video generation job and return the job ID."""
    create_url = f"{endpoint}/openai/v1/video/generations/jobs?api-version={api_version}"
    body = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "n_seconds": n_seconds,
        "model": model
    }
    response = requests.post(create_url, headers=headers, json=body)
    response.raise_for_status()
    print("Full response JSON:", response.json())
    job_id = response.json()["id"]
    print(f"Job created: {job_id}")
    return job_id

def poll_job_status(job_id):
    """Poll for job status until completion and return the final status response."""
    status_url = f"{endpoint}/openai/v1/video/generations/jobs/{job_id}?api-version={api_version}"
    status = None
    status_response = {}
    while status not in ("succeeded", "failed", "cancelled"):
        time.sleep(5)  # Wait before polling again
        status_response = requests.get(status_url, headers=headers).json()
        status = status_response.get("status")
        print(f"Job status: {status}")
    return status_response

def download_video(status_response, output_filename="output.mp4"):
    """Download the generated video if the job succeeded."""
    status = status_response.get("status")
    if status == "succeeded":
        generations = status_response.get("generations", [])
        if generations:
            print("✅ Video generation succeeded.")
            generation_id = generations[0].get("id")
            video_url = f"{endpoint}/openai/v1/video/generations/{generation_id}/content/video?api-version={api_version}"
            video_response = requests.get(video_url, headers=headers)
            if video_response.ok:
                with open(output_filename, "wb") as file:
                    file.write(video_response.content)
                    print(f'Generated video saved as "{output_filename}"')
                return output_filename
        else:
            raise Exception("No generations found in job result.")
    else:
        raise Exception(f"Job didn't succeed. Status: {status}")

def generate_video(prompt, width=1080, height=1080, n_seconds=5, model="sora", output_filename="output.mp4"):
    """Complete video generation workflow: create job, poll status, and download video."""
    # 1. Create a video generation job
    job_id = create_video_job(prompt, width, height, n_seconds, model)

    # 2. Poll for job status
    status_response = poll_job_status(job_id)

    # 3. Retrieve generated video
    return download_video(status_response, output_filename)
