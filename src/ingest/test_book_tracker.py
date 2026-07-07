from src.ingest.book_tracker import BookTracker

tracker = BookTracker()

tracker.process_book({

    "b": "58850.0",
    "a": "58850.5"

})

print(tracker.publish_snapshot())