"""Build a database as it was before events became sequences, with one
small project in it. Prints the project id.

For upgrade-from-events.spec.ts. Run with ADDAXAI_USER_DATA_DIR set to a
throwaway folder and the backend on PYTHONPATH. The rows are written in raw
SQL against the old schema, like the migration tests do
(backend/tests/db/conftest.py), because the models describe the schema at
head.
"""

from sqlalchemy import create_engine

from app.core.config import get_settings
from tests.db.conftest import insert_row, upgrade_to

# The last revision before the rename (b7d9f1a3c5e7).
BEFORE_SEQUENCES = "fa19b0f2a459"

upgrade_to(BEFORE_SEQUENCES)
engine = create_engine(get_settings().database_url)
with engine.begin() as conn:
    project_id = insert_row(
        conn, "projects", name="Before sequences", detection_model_id="MD5A-0-0",
        excluded_classes="[]", counting_threshold=0.5, event_smoothing=1,
        smoothing_strength="normal", taxonomic_rollup=0, independence_interval=1800,
        shortcut_labels="{}", video_fps=3, min_cluster_size=5, min_samples=3,
        timezone="UTC",
    )
    site_id = insert_row(
        conn, "sites", project_id=project_id, name="Pond", latitude=52.0, longitude=5.0,
    )
    deployment_id = insert_row(
        conn, "deployments", project_id=project_id, site_id=site_id,
        folder_path="/cards/pond", folder_status="valid", start_date_local="2024-06-01",
        end_date_local="2024-06-03",
    )

    def event(times: list[str], label: str, max_n: int, **extra: object) -> None:
        """One 30-minute group as the old rule made it, with its count."""
        event_id = insert_row(
            conn, "events", deployment_id=deployment_id, file_count=len(times),
            event_start_local=times[0], event_end_local=times[-1],
            confirmed=extra.pop("confirmed", 0),
        )
        for number, taken in enumerate(times):
            file_id = insert_row(
                conn, "files", deployment_id=deployment_id, file_type="image",
                file_path=f"/cards/pond/{label}_{number}.jpg", captured_at_local=taken,
            )
            insert_row(
                conn, "event_files", event_id=event_id, file_id=file_id,
                sequence_number=number,
            )
            insert_row(
                conn, "detections", file_id=file_id, category="animal", label=label,
                confidence=0.9, verified=0,
            )
        insert_row(
            conn, "event_observations", event_id=event_id, category="animal",
            label=label, max_n=max_n, **extra,
        )

    # Ten minutes apart: one group under the old rule, three sequences under
    # the new one. Somebody confirmed it with a corrected count.
    event(
        ["2024-06-01 10:00:00", "2024-06-01 10:10:00", "2024-06-01 10:20:00"],
        "fox", 2, human_count=3, confirmed=1,
    )
    event(["2024-06-02 14:00:00"], "deer", 1)
engine.dispose()
print(project_id)
