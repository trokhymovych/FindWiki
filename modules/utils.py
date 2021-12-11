from tqdm import tqdm
import requests


class ProgressSession():
    def __init__(self, urls):
        self.pbar = tqdm(total=len(urls), desc='Making async requests')

    def update(self, r, *args, **kwargs):
        if not r.is_redirect:
            self.pbar.update()

    def __enter__(self):
        sess = requests.Session()
        sess.hooks['response'].append(self.update)
        return sess

    def __exit__(self, *args):
        self.pbar.close()