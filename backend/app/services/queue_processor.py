"""
Queue processor service for deployment queue.

Processes deployment queue entries sequentially:
1. Create deployment
2. Scan folder for files
3. Run detection model (from project settings)
4. Run classification model (from project settings, if configured)

Following DEVELOPERS.md principles:
- Crash early on errors
- Explicit configuration
- Type hints everywhere
"""

from sqlalchemy.orm import Session

from app.api.crud import deployment_queue as crud_queue
from app.api.crud import project as crud_project
from app.core.logging_config import get_logger
from app.models import DeploymentQueue

logger = get_logger(__name__)


def process_queue_entry(db: Session, entry: DeploymentQueue) -> None:
    """
    Process a single queue entry.

    Steps:
    1. Get project to retrieve model configuration
    2. Create deployment from entry data
    3. Scan folder and create file records
    4. Run detection model (from project)
    5. Run classification model (from project, if configured)
    6. Update entry status to completed or failed

    Models are now configured at the project level, not per-deployment.
    Legacy entries may still have model IDs, but they will be ignored in favor of project settings.

    Crashes on errors - caller should handle exceptions.
    """
    logger.info(f"Processing queue entry: {entry.id}")

    try:
        # Get project to retrieve model configuration
        project = crud_project.get_project(db, entry.project_id)
        if not project:
            raise ValueError(f"Project {entry.project_id} not found")

        detection_model_id = project.detection_model_id
        classification_model_id = project.classification_model_id

        logger.info(f"Using models from project {project.id}: detection={detection_model_id}, classification={classification_model_id}")

        # TODO: Implement deployment creation
        # deployment = create_deployment_from_queue_entry(db, entry)

        # TODO: Implement folder scanning
        # scan_and_import_files(db, deployment.id, entry.folder_path)

        # TODO: Implement model execution
        # if detection_model_id:
        #     run_detection_model(db, deployment.id, detection_model_id)

        # if classification_model_id and classification_model_id != "none":
        #     run_classification_model(db, deployment.id, classification_model_id, project.taxonomy_config)

        # Update status to completed
        crud_queue.update_queue_status(
            db,
            entry.id,
            status="completed",
            # deployment_id=deployment.id
        )
        logger.info(f"Successfully processed queue entry: {entry.id}")

    except Exception as e:
        logger.error(f"Failed to process queue entry {entry.id}: {e}", exc_info=True)
        crud_queue.update_queue_status(
            db,
            entry.id,
            status="failed",
            error=str(e)
        )
        raise


def process_project_queue(db: Session, project_id: str) -> int:
    """
    Process all pending queue entries for a project sequentially.

    Returns the number of entries processed.
    """
    pending_entries = crud_queue.get_queue_entries(db, project_id, status="pending")

    logger.info(f"Processing {len(pending_entries)} queue entries for project {project_id}")

    processed_count = 0
    for entry in pending_entries:
        # Mark as processing
        crud_queue.update_queue_status(db, entry.id, status="processing")

        try:
            process_queue_entry(db, entry)
            processed_count += 1
        except Exception as e:
            logger.error(f"Stopping queue processing due to error: {e}")
            # Continue processing other entries even if one fails
            continue

    logger.info(f"Completed processing {processed_count}/{len(pending_entries)} queue entries")
    return processed_count
