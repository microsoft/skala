# SPDX-License-Identifier: MIT

import json
import logging
import time
import urllib.request
import uuid
from typing import Any

import qcelemental as qcel
from azure.core.credentials import TokenCredential

from skala.foundry.schemas import Molecule, SkalaConfig, SkalaInput, TaskStatus

LOG = logging.getLogger(__name__)


class SkalaFoundryClient:
    def __init__(self, endpoint: str, credential: TokenCredential | str) -> None:
        """
        Creates a client to interact with a Skala deployment on Azure AI Foundry, based on an endpoint URL and credential token string.
        """
        self._endpoint = endpoint
        self._credential: TokenCredential | str = credential

    def run(
        self,
        molecule: qcel.models.Molecule | Molecule,
        config: SkalaConfig | None = None,
    ) -> TaskStatus:
        if isinstance(molecule, qcel.models.Molecule):
            molecule = Molecule.from_qcel(molecule)
        task = SkalaInput(molecule=molecule, input_config=config or SkalaConfig())
        task_id, deployment_header = self._submit(task)
        try:
            return self._wait_for_task(task_id, deployment_header=deployment_header)
        except (KeyboardInterrupt, Exception):
            self._cancel(task_id, deployment_header=deployment_header)
            raise

    def _wait_for_task(
        self, task_id: uuid.UUID, *, deployment_header: str | None
    ) -> TaskStatus:
        wait_time = 0.5  # seconds
        while True:
            status, _ = self._status(task_id, deployment_header=deployment_header)
            LOG.debug("Status: %s", status)
            if status.status in ("succeeded", "failed", "canceled"):
                return status
            time.sleep(wait_time)
            wait_time *= 1.5  # exponential backoff
            wait_time = min(wait_time, 60)  # seconds

    def _submit(self, task: SkalaInput) -> tuple[uuid.UUID, str | None]:
        data = {"type": "submit", "task": task.model_dump()}
        response, deployment_header = self._request(data)
        if "task_id" not in response:
            raise RuntimeError("No task_id in response")
        return uuid.UUID(response["task_id"]), deployment_header

    def _status(
        self, task_id: uuid.UUID, *, deployment_header: str | None
    ) -> tuple[TaskStatus, str | None]:
        data = {"type": "status", "task_id": str(task_id)}
        response, deployment_header = self._request(
            data, deployment_header=deployment_header
        )
        return TaskStatus.model_validate(response), deployment_header

    def _cancel(self, task_id: uuid.UUID, *, deployment_header: str | None) -> None:
        data = {"type": "cancel", "task_id": str(task_id)}
        LOG.info("Cancelling task %s", task_id)
        self._request(data, deployment_header=deployment_header)

    def _request(
        self, data: dict[str, Any], *, deployment_header: str | None = None
    ) -> tuple[dict[str, Any], str | None]:
        scope = "https://ml.azure.com"
        if isinstance(self._credential, str):
            token = self._credential
        else:
            token = self._credential.get_token(scope).token
        body = str.encode(json.dumps(data))
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }
        # Include deployment header if provided
        if deployment_header is not None:
            headers["azureml-model-deployment"] = deployment_header
        req = urllib.request.Request(self._endpoint, body, headers)
        try:
            response = urllib.request.urlopen(req)
            # Capture deployment header from response for future requests
            response_deployment_header = response.headers.get(
                "azureml-model-deployment"
            )
            result = response.read()
            decoded = json.loads(result.decode("utf-8"))
            return decoded, response_deployment_header
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP error: {e.code} {e.reason}") from e
