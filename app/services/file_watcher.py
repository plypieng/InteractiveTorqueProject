import time
import os
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging

class TorqueFileHandler(FileSystemEventHandler):
    """
    Handles file system events for the torque data directory.
    Updates the shared cache when files are created, deleted, or moved.
    """
    def __init__(self, cache_ref, extension=".csv"):
        self.cache_ref = cache_ref
        self.extension = extension

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(self.extension):
            self._update_cache(event.src_path, "add")

    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith(self.extension):
            self._update_cache(event.src_path, "remove")
            
    def on_moved(self, event):
        if not event.is_directory and event.src_path.endswith(self.extension):
             self._update_cache(event.src_path, "remove")
        if not event.is_directory and event.dest_path.endswith(self.extension):
             self._update_cache(event.dest_path, "add")

    def _update_cache(self, filepath, action):
        filename = os.path.basename(filepath)
        with self.cache_ref["lock"]:
            if action == "add":
                if filename not in self.cache_ref["files"]:
                     self.cache_ref["files"].append(filename)
            elif action == "remove":
                if filename in self.cache_ref["files"]:
                    self.cache_ref["files"].remove(filename)
        logging.info(f"FileSystem Event: {action.upper()} {filename}")

class FileWatcherService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FileWatcherService, cls).__new__(cls)
            cls._instance.observer = Observer()
            cls._instance.watches = {}
            cls._instance.caches = {} # path -> {lock: Lock, files: []}
        return cls._instance

    def start_watching(self, directory_path, extension=".csv", cache_key="default"):
        if not os.path.exists(directory_path):
            logging.warning(f"Watcher cannot start: {directory_path} does not exist.")
            return

        if cache_key in self.caches:
            return # Already watching

        # Initialize cache
        initial_files = [f for f in os.listdir(directory_path) if f.endswith(extension)]
        self.caches[cache_key] = {
            "lock": threading.Lock(),
            "files": initial_files,
            "path": directory_path
        }

        event_handler = TorqueFileHandler(self.caches[cache_key], extension)
        watch = self.observer.schedule(event_handler, directory_path, recursive=False)
        self.watches[cache_key] = watch
        
        if not self.observer.is_alive():
            self.observer.start()
        
        logging.info(f"Started watching {directory_path} for {extension} changes.")

    def get_files(self, cache_key="default"):
        if cache_key not in self.caches:
            return []
        
        # Return a copy to be safe
        with self.caches[cache_key]["lock"]:
            # Need to re-sort because append/remove ruins order? 
            # Or just return list and let caller sort. Caller usually sorts by mtime.
            # Watchdog doesn't track mtime, so we might still need to stat files for sorting 
            # OR we maintain a (filename, mtime) tuple.
            # For now, let's just return the filenames and let the app handle sorting (metadata read is cheaper than listdir).
            return list(self.caches[cache_key]["files"])

    def stop(self):
        self.observer.stop()
        self.observer.join()
