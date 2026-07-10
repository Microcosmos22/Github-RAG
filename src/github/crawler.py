class GithubCrawler:

    def __init__(self, repo_url):
        self.repo_url = repo_url


    def clone_repository(self):
        pass


    def download_issues(self):
        pass


    def download_pull_requests(self):
        pass


crawler = GithubCrawler(
    "https://github.com/pytorch/pytorch"
)

crawler.clone_repository()
crawler.download_issues()

""" Output
data/raw/

repo_files/
    torch/
    aten/
    csrc/

issues.json
prs.json"""
