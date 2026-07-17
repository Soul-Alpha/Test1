from __future__ import annotations

from olympus.core.api_interfaces import ApiRequest, ApiResponse


class HeraApiGateway:
    def handle(self, request: ApiRequest) -> ApiResponse:
        return ApiResponse(ok=True, payload={"route": request.route, "status": "accepted"})
