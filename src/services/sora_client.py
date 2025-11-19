import asyncio

import requests


class SoraClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def submit_job(self, prompt: str):
        response = requests.post(
            f"{self.base_url}/jobs",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"prompt": prompt},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()["job_id"]

    async def poll(self, job_id: str, timeout: int = 120):
        for _ in range(timeout):
            r = requests.get(
                f"{self.base_url}/jobs/{job_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            if data["status"] == "completed":
                return data["video_url"]
            await asyncio.sleep(1)
        raise TimeoutError("Sora job timed out")
