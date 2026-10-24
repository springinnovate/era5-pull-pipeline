"""Run archive publication and catalog requests on one bounded writer thread."""

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
import sqlite3
from threading import Thread
from types import TracebackType
from typing import Any, Concatenate, ParamSpec, TypeVar

from era5_pull_pipeline.catalog import archive_session

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class _CatalogRequest:
    """Pair a catalog operation with its acknowledgement future."""

    operation: Callable[[sqlite3.Connection], Any]
    future: Future[Any]


class PublicationWorker:
    """Own the archive lock and catalog connection on a dedicated writer thread.

    The queue bounds pending catalog requests. The caller separately bounds
    encoded files awaiting publication and retains inputs until acknowledgements.
    Operations return only after their catalog/file contract has completed.
    """

    def __init__(self, archive: Path, capacity: int) -> None:
        """Configure an archive writer with a bounded request queue.

        Args:
            archive: Archive whose lock and connection this worker owns.
            capacity: Maximum waiting requests, excluding the executing request.
        """
        self.archive = archive
        self.requests: Queue[_CatalogRequest | None] = Queue(maxsize=capacity)
        self.started: Future[None] = Future()
        self.error: BaseException | None = None
        self.thread = Thread(target=self._run, name="archive-publication")

    def _run(self) -> None:
        """Execute catalog requests under the archive's single writer lock."""
        try:
            with archive_session(self.archive) as db:
                self.started.set_result(None)
                while True:
                    request = self.requests.get()
                    try:
                        if request is None:
                            return
                        if request.future.set_running_or_notify_cancel():
                            try:
                                result = request.operation(db)
                            except BaseException as exc:
                                request.future.set_exception(exc)
                            else:
                                request.future.set_result(result)
                    finally:
                        self.requests.task_done()
        except BaseException as exc:
            self.error = exc
            if not self.started.done():
                self.started.set_exception(exc)
            self.cancel_pending()

    def __enter__(self) -> "PublicationWorker":
        """Start the writer and wait for archive recovery.

        Returns:
            The running publication worker.

        Raises:
            BaseException: Lock acquisition, catalog opening, or recovery fails.
        """
        self.thread.start()
        try:
            self.started.result()
        except BaseException:
            if self.thread.is_alive():
                self.requests.put(None)
            self.thread.join()
            raise
        return self

    def submit(
        self, operation: Callable[Concatenate[sqlite3.Connection, P], T],
        *args: P.args, **kwargs: P.kwargs,
    ) -> Future[T]:
        """Queue an operation, waiting for request capacity when necessary.

        Args:
            operation: Function called with this worker's connection first.
            *args: Remaining positional arguments.
            **kwargs: Remaining keyword arguments.

        Returns:
            A future carrying the operation's result or original exception.
        """
        future: Future[T] = Future()
        self.requests.put(_CatalogRequest(lambda db: operation(db, *args, **kwargs), future))
        return future

    def try_submit(
        self, operation: Callable[Concatenate[sqlite3.Connection, P], T],
        *args: P.args, **kwargs: P.kwargs,
    ) -> Future[T] | None:
        """Queue an operation without blocking transfer or decoding scheduling.

        Args:
            operation: Function called with this worker's connection first.
            *args: Remaining positional arguments.
            **kwargs: Remaining keyword arguments.

        Returns:
            The acknowledgement future, or None while the queue is full.
        """
        future: Future[T] = Future()
        try:
            self.requests.put_nowait(_CatalogRequest(lambda db: operation(db, *args, **kwargs), future))
        except Full:
            return None
        return future

    def call(
        self, operation: Callable[Concatenate[sqlite3.Connection, P], T],
        *args: P.args, **kwargs: P.kwargs,
    ) -> T:
        """Run a catalog operation and wait for its result outside scheduling loops.

        Args:
            operation: Function called with this worker's connection first.
            *args: Remaining positional arguments.
            **kwargs: Remaining keyword arguments.

        Returns:
            The completed operation's result.

        Raises:
            BaseException: The operation's original failure propagates.
        """
        return self.submit(operation, *args, **kwargs).result()

    def cancel_pending(self) -> None:
        """Cancel requests that have not started after an interrupted pull."""
        while True:
            try:
                request = self.requests.get_nowait()
            except Empty:
                return
            if request is not None:
                request.future.cancel()
            self.requests.task_done()

    def drain(self) -> None:
        """Wait for accepted operations to finish before releasing their inputs."""
        self.requests.join()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Finish the active request before closing the catalog and archive lock.

        Args:
            exc_type: Exception type leaving the caller, if any.
            exc: Original failure, preserved if writer shutdown also fails.
            traceback: Original caller traceback.

        Raises:
            BaseException: Archive shutdown fails without an existing failure.
        """
        if exc is not None:
            self.cancel_pending()
        if self.thread.is_alive():
            self.requests.put(None)
        self.thread.join()
        if self.error is not None:
            if exc is None:
                raise self.error
            exc.add_note(f"Publication worker shutdown failed: {self.error}")
