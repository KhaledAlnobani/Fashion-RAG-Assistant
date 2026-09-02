# src/tracing.py

from phoenix.otel import register
from opentelemetry.trace import Status, StatusCode
from contextlib import contextmanager

tracer_provider = register(
    endpoint="http://localhost:6006/v1/traces",
    project_name="fashion-rag-assistant",
)
tracer = tracer_provider.get_tracer(__name__)


@contextmanager
def trace_span(span_name: str, attributes: dict = None):
    with tracer.start_as_current_span(span_name, attributes=attributes) as span:
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise