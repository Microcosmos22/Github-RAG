"""
Do not add chunking/embeddings yet. Keep this module responsible only for raw data.
The first version of the crawler should do only data acquisition:

1.Clone the repository locally.
2.Walk through source files.
3.Download Github issues.
4.Download pull requests.
"""

import os
import json
import requests

from pathlib import Path
from git import Repo
from dotenv import load_dotenv
from tqdm import tqdm

import subprocess

load_dotenv()


class GithubCrawler:

    def __init__(self, repo_url, output_dir="../../data"):
        self.repo_url = repo_url

        self.output_dir = Path(output_dir)

        self.repo_dir = self.output_dir / "repo_files"
        self.issues_file = self.output_dir / "issues.json"
        self.prs_file = self.output_dir / "prs.json"

        self.token = os.getenv("GITHUB_TOKEN")

        self.headers = {}

        if self.token:
            self.headers["Authorization"] = (f"token {self.token}")


    def clone_repository(self):
        """
        Clone Github repository locally.
        """
        if self.repo_dir.exists():
            print("Repository already exists.")
            return


        self.output_dir.mkdir(parents=True,exist_ok=True)
        print("Cloning repository...")

        subprocess.run(
            ["git", "config", "--global", "core.longpaths", "true"],
            check=True
        )

        try:
            Repo.clone_from(
                self.repo_url,
                self.repo_dir,
                no_checkout=True,
                depth=1
            )
        except Exception as e:
            print(e)
        print(f"Repository cloned to {self.repo_dir}")


    def get_repo_information(self):
        """
        Extract owner/name from URL.
        """

        parts = self.repo_url.rstrip("/").split("/")

        self.owner = parts[-2]
        self.repo = parts[-1]

        return self.owner, self.repo

    def download_issues(self):
        if self.issues_file.exists():
            print("download_issues: issues.json exists already")
            return

        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/issues"

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json"
        }

        issues = []

        with tqdm(desc="Downloading issues") as pbar:
            while url:
                response = requests.get(url, headers=headers,
                    params={"per_page": 100,"state": "open"})

                if response.status_code != 200:
                    raise Exception(response.text)

                batch = response.json()
                issues.extend(batch)

                # Get next page URL from GitHub headers
                url = None
                if "next" in response.links:
                    url = response.links["next"]["url"]

                pbar.update(len(batch))
                pbar.set_postfix(total=len(issues))


        # Exclude pull requests
        issues = [issue for issue in issues if "pull_request" not in issue]
        print(f"Downloaded {len(issues)} issues")

        with open(self.issues_file,"w",encoding="utf-8") as f:
            json.dump(issues,f,indent=2,ensure_ascii=False)

        print(f"Saved {len(issues)} issues")

    def download_pull_requests(self):
        """
        Download Github pull requests.
        """

        owner, repo = self.get_repo_information()

        url = (f"https://api.github.com/repos/"f"{owner}/{repo}/pulls")
        params = {"state": "all","per_page": 100}
        prs = []
        page = 1

        print("Downloading pull requests...")

        while True:

            params["page"] = page
            response = requests.get(url,headers=self.headers,params=params)

            if response.status_code != 200:
                raise Exception(response.text)

            data = response.json()
            batch = response.json()

            if len(data) == 0:
                break

            for pr in data:
                prs.append(
                    {"title": pr["title"],"body": pr["body"],"number": pr["number"],"url": pr["html_url"]})

            page += 1


        with open(self.prs_file,"w",encoding="utf-8") as f:

            json.dump(prs,f,indent=2,ensure_ascii=False)

        print(f"Saved {len(prs)} pull requests")

if __name__ == "__main__":

    crawler = GithubCrawler("https://github.com/pytorch/pytorch")
    crawler.clone_repository()
    owner, repo = crawler.get_repo_information()
    crawler.download_issues()
    crawler.download_pull_requests()
