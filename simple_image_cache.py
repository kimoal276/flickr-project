"""
simple_image_cache.py
---------------------
alternative to nathanael's cache that uses the db
It exposes the exact two methods that `matching.py` actually calls:

    cache.get(url, *, download_missing=False, fast_cache=False, disk_save=False) -> PIL.Image
    cache.get_images(urls, download_missing=False, fast_cache=False, disk_save=False) -> list[Image|None]
"""

from __future__ import annotations

import hashlib
import io
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
from PIL import Image
from tqdm import tqdm

# Tunables
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 20
RETRIES = 3
RETRY_BACKOFF_S = 0.6
NUMBER_OF_THREADS = 32
MAX_MEM_GB = 8                     
DISK_FORMAT = "JPEG"               
DISK_QUALITY = 90                  
MAX_STORE_PX = 1024                
                                   
USER_AGENT = "flico-csv/1.0"


class SimpleImageCache:
    def __init__(
        self,
        cache_dir: str = "image_cache",
        max_mem_gb: float = MAX_MEM_GB,
        n_threads: int = NUMBER_OF_THREADS,
        max_store_px: int | None = MAX_STORE_PX,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_store_px = max_store_px

        self.mem: "OrderedDict[str, tuple[Image.Image, int]]" = OrderedDict()
        self.mem_size_bytes = 0
        self.max_mem_bytes = int(max_mem_gb * 1024 ** 3)
        self.mem_lock = threading.Lock()

        self.n_threads = n_threads

        # thread-local sessions (requests.Session is not guaranteed thread-safe)
        self._tls = threading.local()

    #internal helpers 
    def _session(self) -> requests.Session:
        s = getattr(self._tls, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update({"User-Agent": USER_AGENT})
            self._tls.session = s
        return s

    def _key(self, url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def _path(self, url: str) -> Path:
        ext = ".jpg" if DISK_FORMAT == "JPEG" else ".png"
        return self.cache_dir / f"{self._key(url)}{ext}"

    def _estimate_image_size(self, img: Image.Image) -> int:
        return img.width * img.height * len(img.getbands())

    def _downscale(self, img: Image.Image) -> Image.Image:
        """Cap the longest side to self.max_store_px (only shrinks, never upscales)."""
        if not self.max_store_px:
            return img
        if max(img.width, img.height) <= self.max_store_px:
            return img
        img.thumbnail((self.max_store_px, self.max_store_px), Image.LANCZOS)
        return img

    def _mem_get(self, url: str) -> Optional[Image.Image]:
        with self.mem_lock:
            if url not in self.mem:
                return None
            img, size = self.mem.pop(url)
            self.mem[url] = (img, size)   # move to MRU position
            return img

    def _mem_put(self, url: str, img: Image.Image) -> None:
        size = self._estimate_image_size(img)
        with self.mem_lock:
            if url in self.mem:
                _, old = self.mem.pop(url)
                self.mem_size_bytes -= old
            self.mem[url] = (img, size)
            self.mem_size_bytes += size
            while self.mem_size_bytes > self.max_mem_bytes and self.mem:
                _, (_, evicted) = self.mem.popitem(last=False)
                self.mem_size_bytes -= evicted

    def clear_ram(self) -> None:
        with self.mem_lock:
            self.mem = OrderedDict()
            self.mem_size_bytes = 0

    def _save_disk(self, img: Image.Image, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            if DISK_FORMAT == "JPEG":
                img.save(tmp, format="JPEG", quality=DISK_QUALITY)
            else:
                img.save(tmp, format="PNG")
            tmp.replace(path)            
        except Exception:
            tmp.unlink(missing_ok=True)

    # public API (matches what matching.py expects) 
    def get(
        self,
        url: str,
        *,
        download_missing: bool = False,
        fast_cache: bool = False,
        disk_save: bool = False,
    ) -> Image.Image:
        # 1) RAM
        cached = self._mem_get(url)
        if cached is not None:
            return cached

        # 2) disk
        path = self._path(url)
        if path.exists():
            with path.open("rb") as fh:
                img = Image.open(fh)
                img.load()
            img = img.convert("RGB")
            if fast_cache:
                self._mem_put(url, img)
            return img

        # 3) download
        if not download_missing:
            raise KeyError(f"cache miss: {url}")

        last_exc: Optional[Exception] = None
        for attempt in range(RETRIES):
            try:
                r = self._session().get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content))
                img.load()
                img = img.convert("RGB")
                img = self._downscale(img)
                if disk_save:
                    self._save_disk(img, path)
                if fast_cache:
                    self._mem_put(url, img)
                return img
            except Exception as exc:            
                last_exc = exc
                if attempt < RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_S * (attempt + 1))

        raise RuntimeError(f"download failed for {url}: {last_exc}")

    def get_images(
        self,
        urls: list[str],
        download_missing: bool = False,
        fast_cache: bool = False,
        disk_save: bool = False,
        silent: bool = True,
    ) -> list[Optional[Image.Image]]:
        imgs: list[Optional[Image.Image]] = [None] * len(urls)

        def worker(i: int, url: str):
            try:
                return i, self.get(
                    url,
                    download_missing=download_missing,
                    fast_cache=fast_cache,
                    disk_save=disk_save,
                )
            except Exception:                         
                return i, None

        with ThreadPoolExecutor(max_workers=self.n_threads) as ex:
            futures = [ex.submit(worker, i, u) for i, u in enumerate(urls)]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=("Downloading images" if download_missing else "Reading images"),
                disable=silent,
            ):
                i, img = fut.result()
                imgs[i] = img

        return imgs

    #convenience: bulk pre-download (optional)
    def precache(self, urls, disk_save: bool = True) -> int:
        """Download a list of urls to disk up front. Returns count succeeded."""
        urls = [u for u in dict.fromkeys(map(str, urls)) if u and u != "nan"]
        results = self.get_images(
            urls, download_missing=True, fast_cache=False,
            disk_save=disk_save, silent=False,
        )
        return sum(1 for r in results if r is not None)
