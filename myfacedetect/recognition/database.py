"""
Face Database Management
Tools for managing face recognition databases.
"""
import cv2
import numpy as np
import os
import json
import shutil
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FaceDatabase:
    """Face database management system."""

    def __init__(self, database_path: str = "face_db"):
        self.database_path = database_path
        self.images_path = os.path.join(database_path, "images")
        self.metadata_file = os.path.join(database_path, "metadata.json")

        self._ensure_directories()
        self._load_metadata()

    def _ensure_directories(self):
        """Create database directories if they don't exist."""
        os.makedirs(self.database_path, exist_ok=True)
        os.makedirs(self.images_path, exist_ok=True)

    def _load_metadata(self):
        """Load database metadata."""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {
                    'created_date': datetime.now().isoformat(),
                    'version': '1.0',
                    'people': {}
                }
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            self.metadata = {'people': {}}

    def _save_metadata(self):
        """Save database metadata."""
        try:
            self.metadata['last_modified'] = datetime.now().isoformat()
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def add_person(self, name: str, images: List[np.ndarray],
                   metadata: Optional[Dict] = None) -> bool:
        """
        Add a person to the database.

        Args:
            name: Person's name
            images: List of face images
            metadata: Additional metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create person directory
            person_dir = os.path.join(self.images_path, name)
            os.makedirs(person_dir, exist_ok=True)

            # Save images
            saved_files = []
            for i, image in enumerate(images):
                filename = f"{name}_{i:03d}.jpg"
                filepath = os.path.join(person_dir, filename)
                cv2.imwrite(filepath, image)
                saved_files.append(filename)

            # Update metadata
            if name not in self.metadata['people']:
                self.metadata['people'][name] = {
                    'added_date': datetime.now().isoformat(),
                    'images': [],
                    'metadata': metadata or {}
                }

            self.metadata['people'][name]['images'].extend(saved_files)
            self.metadata['people'][name]['last_updated'] = datetime.now(
            ).isoformat()

            self._save_metadata()

            logger.info(f"Added {len(images)} images for {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to add person {name}: {e}")
            return False

    def get_person_images(self, name: str) -> List[np.ndarray]:
        """
        Get all images for a person.

        Args:
            name: Person's name

        Returns:
            List of images
        """
        try:
            if name not in self.metadata['people']:
                return []

            images = []
            person_dir = os.path.join(self.images_path, name)

            for filename in self.metadata['people'][name]['images']:
                filepath = os.path.join(person_dir, filename)
                if os.path.exists(filepath):
                    image = cv2.imread(filepath)
                    if image is not None:
                        images.append(image)

            return images

        except Exception as e:
            logger.error(f"Failed to get images for {name}: {e}")
            return []

    def remove_person(self, name: str) -> bool:
        """
        Remove a person from the database.

        Args:
            name: Person's name

        Returns:
            True if successful, False otherwise
        """
        try:
            if name not in self.metadata['people']:
                return False

            # Remove person directory
            person_dir = os.path.join(self.images_path, name)
            if os.path.exists(person_dir):
                shutil.rmtree(person_dir)

            # Remove from metadata
            del self.metadata['people'][name]
            self._save_metadata()

            logger.info(f"Removed {name} from database")
            return True

        except Exception as e:
            logger.error(f"Failed to remove {name}: {e}")
            return False

    def list_people(self) -> List[str]:
        """Get list of all people in database."""
        return list(self.metadata['people'].keys())

    def get_person_metadata(self, name: str) -> Optional[Dict]:
        """Get metadata for a person."""
        return self.metadata['people'].get(name)

    def update_person_metadata(self, name: str, metadata: Dict) -> bool:
        """Update metadata for a person."""
        try:
            if name not in self.metadata['people']:
                return False

            self.metadata['people'][name]['metadata'].update(metadata)
            self.metadata['people'][name]['last_updated'] = datetime.now(
            ).isoformat()
            self._save_metadata()

            return True
        except Exception as e:
            logger.error(f"Failed to update metadata for {name}: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {
            'total_people': len(self.metadata['people']),
            'total_images': sum(len(person['images']) for person in self.metadata['people'].values()),
            'database_path': self.database_path,
            'created_date': self.metadata.get('created_date'),
            'last_modified': self.metadata.get('last_modified')
        }

        if self.metadata['people']:
            images_per_person = [len(person['images'])
                                 for person in self.metadata['people'].values()]
            stats['avg_images_per_person'] = np.mean(images_per_person)
            stats['min_images_per_person'] = min(images_per_person)
            stats['max_images_per_person'] = max(images_per_person)

        return stats

    def export_database(self, export_path: str) -> bool:
        """Export database to a directory."""
        try:
            shutil.copytree(self.database_path,
                            export_path, dirs_exist_ok=True)
            logger.info(f"Database exported to {export_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export database: {e}")
            return False

    def import_database(self, import_path: str, merge: bool = False) -> bool:
        """Import database from a directory."""
        try:
            if not merge:
                # Clear existing database
                if os.path.exists(self.database_path):
                    shutil.rmtree(self.database_path)
                shutil.copytree(import_path, self.database_path)
            else:
                # Merge databases
                import_metadata_file = os.path.join(
                    import_path, "metadata.json")
                if os.path.exists(import_metadata_file):
                    with open(import_metadata_file, 'r') as f:
                        import_metadata = json.load(f)

                    # Copy images
                    import_images_path = os.path.join(import_path, "images")
                    if os.path.exists(import_images_path):
                        for person_name in import_metadata.get('people', {}):
                            person_src = os.path.join(
                                import_images_path, person_name)
                            person_dst = os.path.join(
                                self.images_path, person_name)
                            if os.path.exists(person_src):
                                shutil.copytree(
                                    person_src, person_dst, dirs_exist_ok=True)

                    # Merge metadata
                    for name, data in import_metadata.get('people', {}).items():
                        if name not in self.metadata['people']:
                            self.metadata['people'][name] = data
                        else:
                            # Merge images lists
                            existing_images = set(
                                self.metadata['people'][name]['images'])
                            new_images = [
                                img for img in data['images'] if img not in existing_images]
                            self.metadata['people'][name]['images'].extend(
                                new_images)

            self._load_metadata()
            logger.info(f"Database imported from {import_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to import database: {e}")
            return False

    def clean_database(self) -> Dict[str, int]:
        """Clean database by removing missing files and fixing inconsistencies."""
        stats = {'removed_files': 0, 'fixed_metadata': 0}

        try:
            for name in list(self.metadata['people'].keys()):
                person_dir = os.path.join(self.images_path, name)
                if not os.path.exists(person_dir):
                    # Person directory missing, remove from metadata
                    del self.metadata['people'][name]
                    stats['fixed_metadata'] += 1
                    continue

                # Check each image file
                valid_images = []
                for filename in self.metadata['people'][name]['images']:
                    filepath = os.path.join(person_dir, filename)
                    if os.path.exists(filepath):
                        valid_images.append(filename)
                    else:
                        stats['removed_files'] += 1

                # Update image list
                if len(valid_images) != len(self.metadata['people'][name]['images']):
                    self.metadata['people'][name]['images'] = valid_images
                    stats['fixed_metadata'] += 1

                # Remove person if no valid images
                if not valid_images:
                    del self.metadata['people'][name]
                    if os.path.exists(person_dir):
                        shutil.rmtree(person_dir)
                    stats['fixed_metadata'] += 1

            self._save_metadata()
            logger.info(f"Database cleaned: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Failed to clean database: {e}")
            return stats


def create_face_database(database_path: str = "face_db") -> FaceDatabase:
    """
    Create a face database instance.

    Args:
        database_path: Path to database directory

    Returns:
        FaceDatabase instance
    """
    return FaceDatabase(database_path)
