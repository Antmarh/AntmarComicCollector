# antmar/update.py
import json, urllib.request

def get_latest_tag(user="Antmarh", repo="AntmarComicCollector"):
    url = f"https://api.github.com/repos/{user}/{repo}/releases/latest"
    with urllib.request.urlopen(url, timeout=5) as r:
        data = json.load(r)
    return data.get("tag_name","")

def is_newer(cur:str, latest:str)->bool:
    def norm(v): return [int(x) for x in v.strip("v").split(".")]
    try: return norm(latest) > norm(cur)
    except: return False
