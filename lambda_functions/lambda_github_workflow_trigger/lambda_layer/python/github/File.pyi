from typing import Any, Dict

from github.GithubObject import NonCompletableGithubObject

class File(NonCompletableGithubObject):
    def __repr__(self) -> str: ...
    def _initAttributes(self) -> None: ...
    def _useAttributes(self, attributes: Dict[str, Any]) -> None: ...
    @property
    def additions(self) -> int: ...
    @property
    def blob_url(self) -> str: ...
    @property
    def changes(self) -> int: ...
    @property
    def contents_url(self) -> str: ...
    @property
    def deletions(self) -> int: ...
    @property
    def filename(self) -> str: ...
    @property
    def patch(self) -> str: ...
    @property
    def raw_url(self) -> str: ...
    @property
    def sha(self) -> str: ...
    @property
    def status(self) -> str: ...
    @property
    def previous_filename(self) -> str: ...
